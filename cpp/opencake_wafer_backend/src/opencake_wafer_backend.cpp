#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct HardwareConfig {
  int ranks = 0;
  double compute_ops_per_cycle_per_rank = 0.0;
  double sram_bandwidth_bytes_per_cycle_per_rank = 0.0;
  double sram_bytes_per_rank = 0.0;
  double link_bandwidth_bytes_per_cycle = 0.0;
  double router_latency_cycles = 0.0;
};

struct Result {
  int ranks = 0;
  long long compute_ops = 0;
  double compute_cycles = 0.0;
  double memory_cycles = 0.0;
  double network_cycles = 0.0;
  double sim_time_cycles = 0.0;
  long long max_sram_bytes_per_rank = 0;
  double max_sram_utilization = 0.0;
  double max_sram_pressure_ratio = 0.0;
  bool sram_capacity_feasible = true;
  double max_link_utilization = 0.0;
  double compute_efficiency = 0.0;
};

using Coord = std::pair<int, int>;

static std::string tensor_key(const Coord& rank, const std::string& tensor, int k_start) {
  return std::to_string(rank.first) + "," + std::to_string(rank.second) + ":" + tensor + ":" +
         std::to_string(k_start);
}

static std::string read_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

static double json_number(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("missing JSON number key " + key);
  }
  return std::stod(match[1].str());
}

static std::string json_string(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("missing JSON string key " + key);
  }
  return match[1].str();
}

static std::string json_string_or(
    const std::string& text,
    const std::string& key,
    const std::string& fallback) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return fallback;
  }
  return match[1].str();
}

static double json_number_or(const std::string& text, const std::string& key, double fallback) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return fallback;
  }
  return std::stod(match[1].str());
}

static double yml_bracket_number(const std::string& text, const std::string& key) {
  const std::regex pattern(key + "\\s*:\\s*\\[\\s*([-+0-9.eE]+)\\s*\\]");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("missing YAML bracket number key " + key);
  }
  return std::stod(match[1].str());
}

static Coord json_coord(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("missing coord key " + key);
  }
  return {std::stoi(match[1].str()), std::stoi(match[2].str())};
}

static std::vector<Coord> json_route(const std::string& line) {
  const std::string key = "\"route\"";
  const std::size_t key_pos = line.find(key);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("send record missing route");
  }
  const std::size_t start = line.find("[[", key_pos);
  const std::size_t end = line.find("]]", start);
  if (start == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("invalid route array");
  }
  const std::string route_text = line.substr(start, end - start + 2);
  const std::regex pair_pattern("\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
  std::vector<Coord> route;
  for (std::sregex_iterator it(route_text.begin(), route_text.end(), pair_pattern), last;
       it != last;
       ++it) {
    route.push_back({std::stoi((*it)[1].str()), std::stoi((*it)[2].str())});
  }
  return route;
}

static HardwareConfig read_hardware(const std::string& package_root) {
  const std::string system = read_file(package_root + "/system.json");
  const std::string network = read_file(package_root + "/network.yml");
  HardwareConfig config;
  config.ranks = static_cast<int>(json_number(system, "ranks"));
  config.compute_ops_per_cycle_per_rank = json_number(system, "compute_ops_per_cycle_per_rank");
  config.sram_bandwidth_bytes_per_cycle_per_rank =
      json_number(system, "sram_bandwidth_bytes_per_cycle_per_rank");
  config.sram_bytes_per_rank = json_number(system, "sram_bytes_per_rank");
  config.link_bandwidth_bytes_per_cycle =
      yml_bracket_number(network, "link_bandwidth_bytes_per_cycle");
  config.router_latency_cycles = yml_bracket_number(network, "router_latency_cycles");
  return config;
}

static Result run_backend(const std::string& package_root, const HardwareConfig& config) {
  std::ifstream spatial_ir(package_root + "/workload/opencake_spatial_ir.jsonl");
  if (!spatial_ir) {
    throw std::runtime_error("cannot open spatial IR");
  }

  std::map<Coord, double> compute_ready_by_rank;
  std::map<Coord, double> memory_ready_by_rank;
  std::map<Coord, double> compute_busy_by_rank;
  std::map<Coord, double> memory_busy_by_rank;
  std::map<Coord, long long> sram_bytes_by_rank;
  std::map<std::pair<Coord, Coord>, double> link_busy_cycles;
  std::map<std::pair<Coord, Coord>, double> link_ready_cycles;
  std::map<std::string, double> tensor_ready_by_rank;
  long long total_compute_ops = 0;
  double last_end_cycle = 0.0;

  std::string line;
  while (std::getline(spatial_ir, line)) {
    if (line.empty()) {
      continue;
    }
    const std::string op = json_string(line, "op");
    if (op == "alloc") {
      const Coord rank = json_coord(line, "rank");
      const auto bytes = static_cast<long long>(json_number(line, "bytes"));
      sram_bytes_by_rank[rank] += bytes;
    } else if (op == "compute") {
      const Coord rank = json_coord(line, "rank");
      const auto ops = static_cast<long long>(json_number(line, "ops"));
      const double memory_bytes = json_number(line, "memory_bytes");
      const int k_start = static_cast<int>(json_number_or(line, "k_start", 0));
      total_compute_ops += ops;
      const double a_ready = tensor_ready_by_rank[tensor_key(rank, "A", k_start)];
      const double b_ready = tensor_ready_by_rank[tensor_key(rank, "B", k_start)];
      const double memory_start = std::max({a_ready, b_ready, memory_ready_by_rank[rank]});
      const double memory_duration = memory_bytes / config.sram_bandwidth_bytes_per_cycle_per_rank;
      const double memory_end = memory_start + memory_duration;
      memory_ready_by_rank[rank] = memory_end;
      memory_busy_by_rank[rank] += memory_duration;

      const double compute_start = std::max(memory_end, compute_ready_by_rank[rank]);
      const double compute_duration = ops / config.compute_ops_per_cycle_per_rank;
      const double compute_end = compute_start + compute_duration;
      compute_ready_by_rank[rank] = compute_end;
      compute_busy_by_rank[rank] += compute_duration;
      last_end_cycle = std::max(last_end_cycle, std::max(memory_end, compute_end));
    } else if (op == "send") {
      const Coord src_rank = json_coord(line, "src_rank");
      const Coord dst_rank = json_coord(line, "dst_rank");
      const std::string tensor = json_string_or(line, "tensor", "");
      const int k_start = static_cast<int>(json_number_or(line, "k_start", 0));
      const double bytes = json_number(line, "bytes");
      const std::vector<Coord> route = json_route(line);
      const double per_link_cycles =
          config.router_latency_cycles + bytes / config.link_bandwidth_bytes_per_cycle;
      double arrival_cycle = tensor_ready_by_rank[tensor_key(src_rank, tensor, k_start)];
      for (std::size_t index = 0; index + 1 < route.size(); ++index) {
        const auto link = std::make_pair(route[index], route[index + 1]);
        const double start_cycle = std::max(arrival_cycle, link_ready_cycles[link]);
        const double end_cycle = start_cycle + per_link_cycles;
        link_ready_cycles[link] = end_cycle;
        link_busy_cycles[link] += per_link_cycles;
        arrival_cycle = end_cycle;
      }
      if (!tensor.empty()) {
        tensor_ready_by_rank[tensor_key(dst_rank, tensor, k_start)] = arrival_cycle;
      }
      last_end_cycle = std::max(last_end_cycle, arrival_cycle);
    } else if (op == "define_color") {
      continue;
    } else {
      throw std::runtime_error("unsupported Spatial IR op " + op);
    }
  }

  Result result;
  result.ranks = config.ranks;
  result.compute_ops = total_compute_ops;
  for (const auto& item : compute_busy_by_rank) {
    result.compute_cycles = std::max(result.compute_cycles, item.second);
  }
  for (const auto& item : memory_busy_by_rank) {
    result.memory_cycles = std::max(result.memory_cycles, item.second);
  }
  for (const auto& item : link_busy_cycles) {
    result.network_cycles = std::max(result.network_cycles, item.second);
  }
  for (const auto& item : sram_bytes_by_rank) {
    result.max_sram_bytes_per_rank = std::max(result.max_sram_bytes_per_rank, item.second);
  }
  result.sim_time_cycles = last_end_cycle;
  result.max_sram_pressure_ratio = result.max_sram_bytes_per_rank / config.sram_bytes_per_rank;
  result.max_sram_utilization = std::min(result.max_sram_pressure_ratio, 1.0);
  result.sram_capacity_feasible = result.max_sram_pressure_ratio <= 1.0;
  result.max_link_utilization =
      result.sim_time_cycles == 0.0 ? 0.0 : result.network_cycles / result.sim_time_cycles;
  const double peak_compute =
      result.sim_time_cycles * config.compute_ops_per_cycle_per_rank * config.ranks;
  result.compute_efficiency = peak_compute == 0.0 ? 0.0 : result.compute_ops / peak_compute;
  return result;
}

static void print_result(const Result& result) {
  std::cout << "{\n"
            << "  \"backend\": \"opencake_wafer_backend_cpp\",\n"
            << "  \"compute_cycles\": " << result.compute_cycles << ",\n"
            << "  \"compute_efficiency\": " << result.compute_efficiency << ",\n"
            << "  \"compute_ops\": " << result.compute_ops << ",\n"
            << "  \"max_link_utilization\": " << result.max_link_utilization << ",\n"
            << "  \"max_sram_bytes_per_rank\": " << result.max_sram_bytes_per_rank << ",\n"
            << "  \"max_sram_pressure_ratio\": " << result.max_sram_pressure_ratio << ",\n"
            << "  \"max_sram_utilization\": " << result.max_sram_utilization << ",\n"
            << "  \"memory_cycles\": " << result.memory_cycles << ",\n"
            << "  \"network_cycles\": " << result.network_cycles << ",\n"
            << "  \"ranks\": " << result.ranks << ",\n"
            << "  \"sim_time_cycles\": " << result.sim_time_cycles << ",\n"
            << "  \"sram_capacity_feasible\": "
            << (result.sram_capacity_feasible ? "true" : "false") << "\n"
            << "}\n";
}

static void check_reference(const std::string& package_root, const Result& result) {
  const std::string reference = read_file(package_root + "/reference_result.json");
  const double tolerance = 1e-9;
  const std::vector<std::pair<std::string, double>> checks = {
      {"compute_cycles", result.compute_cycles},
      {"memory_cycles", result.memory_cycles},
      {"network_cycles", result.network_cycles},
      {"sim_time_cycles", result.sim_time_cycles},
      {"max_sram_utilization", result.max_sram_utilization},
      {"max_sram_pressure_ratio", result.max_sram_pressure_ratio},
      {"max_link_utilization", result.max_link_utilization},
      {"compute_efficiency", result.compute_efficiency},
  };
  for (const auto& check : checks) {
    const double expected = json_number(reference, check.first);
    if (std::abs(expected - check.second) > tolerance) {
      throw std::runtime_error(
          "reference mismatch for " + check.first + ": expected " +
          std::to_string(expected) + ", got " + std::to_string(check.second));
    }
  }
  const auto expected_ops = static_cast<long long>(json_number(reference, "compute_ops"));
  if (expected_ops != result.compute_ops) {
    throw std::runtime_error("reference mismatch for compute_ops");
  }
}

int main(int argc, char** argv) {
  try {
    std::string package_root;
    bool check = false;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--package" && index + 1 < argc) {
        package_root = argv[++index];
      } else if (arg == "--check-reference") {
        check = true;
      } else {
        throw std::runtime_error("usage: opencake_wafer_backend --package DIR [--check-reference]");
      }
    }
    if (package_root.empty()) {
      throw std::runtime_error("--package is required");
    }
    const HardwareConfig config = read_hardware(package_root);
    const Result result = run_backend(package_root, config);
    if (check) {
      check_reference(package_root, result);
    }
    print_result(result);
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
