ThisBuild / scalaVersion := "2.13.12"
ThisBuild / organization := "edu.bu.wafersim"
ThisBuild / version := "0.1.0"

val chiselVersion = "6.5.0"

lazy val root = (project in file("."))
  .settings(
    name := "wafersim-rtl",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "edu.berkeley.cs" %% "chiseltest" % "6.0.0" % Test,
      "org.scalatest" %% "scalatest" % "3.2.18" % Test
    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
    )
  )

