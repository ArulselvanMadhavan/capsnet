type t =
  { batch_size : int
  ; test_batch_size : int
  ; epochs : int
  ; lr : float
  ; no_cuda : bool
  ; seed : int
  ; log_interval : int
  ; routing_iterations : int
  ; with_reconstruction : bool
  }

let squash x =
  let open Torch in
  let l2 = Tensor.pow_tensor_scalar x ~exponent:(Scalar.i 2) in
  let l2 = Tensor.sum_dim_intlist l2 ~dim:(Some [ 2 ]) ~keepdim:false ~dtype:(T Float) in
  let lengths = Tensor.sqrt l2 in
  let l2 = Tensor.div l2 (Tensor.add_scalar l2 (Scalar.i 1)) in
  Tensor.mul l2 (Tensor.div x lengths)
;;

module PrimaryCapsLayer = struct
  type t =
    { conv : Torch.Nn.t
    ; output_caps : int
    ; output_dim : int
    ; input_channels : int
    }

  let make vs ~input_channels ~output_caps ~output_dim ~kernel_size ~stride =
    let conv =
      Torch.Layer.conv2d
        vs
        ~ksize:(kernel_size, kernel_size)
        ~stride:(stride, stride)
        ~input_dim:input_channels
        (output_caps * output_dim)
    in
    { conv; output_caps; output_dim; input_channels }
  ;;

  let forward t input =
    let open Torch in
    let out = Layer.forward t.conv input in
    let out_shp = Array.of_list (Tensor.shape out) in
    let out =
      Tensor.view
        out
        ~size:[ out_shp.(0); t.output_caps; t.output_dim; out_shp.(2); out_shp.(3) ]
    in
    let out = Tensor.permute out ~dims:[ 0; 1; 3; 4; 2 ] in
    let out = Tensor.contiguous out in
    let out_shp = Array.of_list (Tensor.shape out) in
    let out = Tensor.view out ~size:[ out_shp.(0); -1; out_shp.(4) ] in
    squash out
  ;;
end

module CapsNet = struct
  type t = { conv1 : Torch.Nn.t }

  let make vs ~routing_iterations =
    let conv1 = Torch.Layer.conv2d vs ~ksize:(9, 9) ~stride:(1, 1) ~input_dim:1 256 in
    (* let primary_caps = PrimaryCaps.make *)
    { conv1 }
  ;;
end

let make
  ?(batch_size = 128)
  ?(test_batch_size = 1000)
  ?(epochs = 250)
  ?(lr = 0.001)
  ?(no_cuda = false)
  ?(seed = 1)
  ?(log_interval = 10)
  ?(routing_iterations = 3)
  ?(with_reconstruction = false)
  ()
  =
  let no_cuda = (not no_cuda) && not (Torch.Cuda.is_available ()) in
  Torch_core.Wrapper.manual_seed seed;
  let _mnist = Torch.Mnist_helper.read_files () in
  let model = routing_iterations in
  { batch_size
  ; test_batch_size
  ; epochs
  ; lr
  ; no_cuda
  ; seed
  ; log_interval
  ; routing_iterations
  ; with_reconstruction
  }
;;
