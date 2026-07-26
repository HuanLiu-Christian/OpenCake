module {
  func.func @main(
      %A: tensor<1024x1024xf32>,
      %B: tensor<1024x1024xf32>,
      %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = linalg.matmul
        ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%C : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}
