let squash x =
  let open Torch in
  let l2 = Tensor.pow_tensor_scalar x ~exponent:(Scalar.i 2) in
  let l2 = Tensor.sum_dim_intlist l2 ~dim:(Some [ 2 ]) ~keepdim:false ~dtype:(T Float) in
  let lengths = Tensor.sqrt l2 in
  let l2 = Tensor.div l2 (Tensor.add_scalar l2 (Scalar.i 1)) in
  let l2 = Tensor.div l2 lengths in
  let x_shp = Array.of_list (Tensor.shape x) in
  Tensor.mul x (Tensor.view l2 ~size:[ x_shp.(0); x_shp.(1); 1 ])
;;

module AgreementRouting = struct
  open Torch

  type t =
    { n_iterations : int
    ; b : Tensor.t
    }

  let make vs ~input_caps ~output_caps ~n_iterations =
    let b =
      Var_store.new_var
        ~trainable:true
        vs
        ~shape:[ input_caps; output_caps ]
        ~init:Zeros
        ~name:"b"
    in
    { n_iterations; b }
  ;;

  let forward t u_predict =
    (* c = softmax(b); s = c * u_pred; v = squash s; agreement = v * u_pred; b = b + agreement*)
    let c = Tensor.softmax t.b ~dim:(-1) ~dtype:(T Float) in
    let c = Tensor.unsqueeze c ~dim:2 in
    (* s = c * u *)
    let s = Tensor.mul c u_predict in
    let s = Tensor.sum_dim_intlist s ~dim:(Some [ 1 ]) ~keepdim:false ~dtype:(T Float) in
    (* capsule output *)
    let v = squash s in
    let v = ref v in
    for _i = 0 to t.n_iterations - 1 do
      let sizes = Array.of_list (Tensor.size u_predict) in
      let batch_size = sizes.(0) in
      let input_caps = sizes.(1) in
      let output_caps = sizes.(2) in
      let b_batch =
        Tensor.expand t.b ~size:[ batch_size; input_caps; output_caps ] ~implicit:true
      in
      v := Tensor.unsqueeze !v ~dim:1;
      let agreement =
        Tensor.sum_dim_intlist
          (Tensor.mul u_predict !v)
          ~dim:(Some [ -1 ])
          ~dtype:(T Float)
          ~keepdim:false
      in
      let b_batch = Tensor.add b_batch agreement in
      let c =
        Tensor.softmax
          (Tensor.view b_batch ~size:[ -1; output_caps ])
          ~dim:(-1)
          ~dtype:(T Float)
      in
      let c = Tensor.view c ~size:[ -1; input_caps; output_caps; 1 ] in
      (* s = c * u *)
      let s = Tensor.mul c u_predict in
      let s =
        Tensor.sum_dim_intlist s ~dim:(Some [ 1 ]) ~dtype:(T Float) ~keepdim:false
      in
      v := squash s
    done;
    !v
  ;;
end

module CapsLayer = struct
  open Torch

  type t =
    { weights : Tensor.t
    ; routing_module : AgreementRouting.t
    ; input_caps : int
    ; output_caps : int
    ; output_dim : int
    }

  let make vs ~input_caps ~input_dim ~output_caps ~output_dim ~routing_module =
    let weights =
      Var_store.new_var
        vs
        ~shape:[ input_caps; input_dim; output_caps * output_dim ]
        ~name:"weights"
        ~init:(Normal { mean = 0.; stdev = 1. })
    in
    { weights; routing_module; input_caps; output_caps; output_dim }
  ;;

  let forward t caps_output =
    let caps_output = Tensor.unsqueeze caps_output ~dim:2 in
    Stdio.printf "%s|%s\n" (Tensor.shape_str caps_output) (Tensor.shape_str t.weights);
    let u_predict = Tensor.matmul caps_output t.weights in
    let u_predict =
      Tensor.view
        u_predict
        ~size:
          [ List.hd (Tensor.shape u_predict); t.input_caps; t.output_caps; t.output_dim ]
    in
    AgreementRouting.forward t.routing_module u_predict
  ;;
end

module PrimaryCapsLayer = struct
  type t =
    { conv : Torch.Nn.t
    ; output_caps : int
    ; output_dim : int
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
    { conv; output_caps; output_dim }
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

(* module type Net = sig *)
(*   open Torch *)

(*   type t *)

(*   val forward : t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t *)
(* end *)

(* module Model (N : Net) = struct *)
(*   type t = N.t *)

(*   let forward = N.forward *)
(* end *)

module CapsNet = struct
  open Torch

  type t =
    { conv1 : Torch.Nn.t
    ; primary_caps : PrimaryCapsLayer.t
    ; digit_caps : CapsLayer.t
    }

  let make vs routing_iterations =
    let conv1 = Torch.Layer.conv2d vs ~ksize:(9, 9) ~stride:(1, 1) ~input_dim:1 256 in
    let primary_caps =
      PrimaryCapsLayer.make
        vs
        ~input_channels:256
        ~output_caps:32
        ~output_dim:8
        ~kernel_size:9
        ~stride:2
    in
    let input_caps = 32 * 6 * 6 in
    let num_classes = 10 in
    let routing_module =
      AgreementRouting.make
        vs
        ~input_caps
        ~output_caps:num_classes
        ~n_iterations:routing_iterations
    in
    let digit_caps =
      CapsLayer.make
        vs
        ~input_caps
        ~input_dim:8
        ~output_caps:num_classes
        ~output_dim:16
        ~routing_module
    in
    { conv1; primary_caps; digit_caps }
  ;;

  let forward t input _ =
    let x = Layer.forward t.conv1 input in
    let x = Tensor.relu x in
    let x = PrimaryCapsLayer.forward t.primary_caps x in
    let x = CapsLayer.forward t.digit_caps x in
    let probs = Tensor.pow_tensor_scalar x ~exponent:(Scalar.i 2) in
    let probs =
      Tensor.sum_dim_intlist probs ~dim:(Some [ 2 ]) ~keepdim:false ~dtype:(T Float)
    in
    x, probs
  ;;
end

module ReconstructionNet = struct
  open Torch

  type t =
    { fc1 : Nn.t
    ; fc2 : Nn.t
    ; fc3 : Nn.t
    ; n_dim : int
    ; n_classes : int
    }

  let make vs n_dim n_classes =
    let fc1 = Layer.linear vs ~input_dim:(n_dim * n_classes) 512 in
    let fc2 = Layer.linear vs ~input_dim:512 1024 in
    let fc3 = Layer.linear vs ~input_dim:1024 784 in
    { fc1; fc2; fc3; n_dim; n_classes }
  ;;

  let forward t x target =
    let mask = Tensor.zeros [ List.hd (Tensor.shape x); t.n_classes ] in
    let mask =
      Tensor.scatter_
        mask
        ~dim:1
        ~index:(Tensor.view target ~size:[ -1; 1 ])
        ~src:
          (Tensor.scalar_tensor ~s:(Scalar.f 1.) ~options:(T Float, Tensor.device target))
    in
    let mask = Tensor.unsqueeze mask ~dim:2 in
    let x = Tensor.mul x mask in
    let x = Tensor.view x ~size:[ -1; t.n_dim * t.n_classes ] in
    let x = Tensor.relu (Layer.forward t.fc1 x) in
    let x = Tensor.relu (Layer.forward t.fc2 x) in
    let x = Tensor.sigmoid (Layer.forward t.fc3 x) in
    x
  ;;
end

module CapsNetWithReconstruction = struct
  type t =
    { capsnet : CapsNet.t
    ; reconstruction_net : ReconstructionNet.t
    }

  let make capsnet reconstruction_net = { capsnet; reconstruction_net }

  let forward t x target =
    let x, probs =
      CapsNet.forward
        t.capsnet
        x
        (Torch.Tensor.empty ~size:[] ~options:(T Float, Torch.Tensor.device x))
    in
    let reconstruction = ReconstructionNet.forward t.reconstruction_net x target in
    reconstruction, probs
  ;;
end

module MarginLoss = struct
  open Torch

  type t =
    { m_pos : float
    ; m_neg : float
    ; lambda_ : float
    }

  let make m_pos m_neg lambda_ = { m_pos; m_neg; lambda_ }

  let forward self lengths targets size_average =
    let t = Tensor.zeros (Tensor.size lengths) in
    let t = Tensor.to_device t ~device:(Tensor.device targets) in
    let t =
      Tensor.scatter_
        t
        ~dim:1
        ~index:(Tensor.view targets ~size:[ -1; 1 ])
        ~src:
          (Tensor.scalar_tensor ~s:(Scalar.i 1) ~options:(T Float, Tensor.device targets))
    in
    let targets = t in
    let diff =
      Tensor.pow_tensor_scalar
        (Tensor.relu (Tensor.add_scalar (Tensor.neg lengths) (Scalar.f self.m_pos)))
        ~exponent:(Scalar.i 2)
    in
    let pos_loss = Tensor.mul targets diff in
    let diff =
      Tensor.pow_tensor_scalar
        (Tensor.relu (Tensor.sub_scalar lengths (Scalar.f self.m_neg)))
        ~exponent:(Scalar.i 2)
    in
    let neg_loss =
      Tensor.mul_scalar
        (Tensor.add_scalar (Tensor.neg targets) (Scalar.f 1.))
        (Scalar.f self.lambda_)
    in
    let neg_loss = Tensor.mul diff neg_loss in
    let losses = Tensor.add pos_loss neg_loss in
    if size_average then Tensor.mean losses else Tensor.sum losses
  ;;
end

type t =
  { batch_size : int
      (* ; test_batch_size : int *)
      (* ; epochs : int *)
      (* ; lr : float *)
      (* ; no_cuda : bool *)
      (* ; seed : int *)
      (* ; log_interval : int *)
      (* ; routing_iterations : int *)
  ; with_reconstruction : bool (* ; capsmodel : CapsNet.t *)
  ; recmodel : CapsNetWithReconstruction.t
  ; adam : Torch.Optimizer.t
  ; loss : MarginLoss.t
  ; mnist : Torch.Dataset_helper.t
  }

let make
  ?(batch_size = 128)
  ?((* ?(test_batch_size = 1000) *)
    (* ?(epochs = 250) *)
    lr = 0.001)
  ?(no_cuda = false)
  ?(seed = 1)
  ?((* ?(log_interval = 10) *)
    routing_iterations = 3)
  ?(with_reconstruction = false)
  ()
  =
  let open Torch in
  let no_cuda = (not no_cuda) && not (Torch.Cuda.is_available ()) in
  Torch_core.Wrapper.manual_seed seed;
  let mnist = Mnist_helper.read_files () in
  let device = if no_cuda then Device.Cpu else Device.Cuda 0 in
  let vs = Var_store.create ~name:"capsnet" ~device () in
  let capsmodel = CapsNet.make vs routing_iterations in
  let recmodel =
    CapsNetWithReconstruction.make capsmodel (ReconstructionNet.make vs 16 10)
  in
  let adam = Optimizer.adam vs ~learning_rate:lr in
  let loss = MarginLoss.make 0.9 0.1 0.5 in
  { batch_size
    (* ; test_batch_size *)
    (* ; epochs *)
    (* ; lr *)
    (* ; no_cuda *)
    (* ; seed *)
    (* ; log_interval *)
    (* ; routing_iterations *)
  ; with_reconstruction (* ; capsmodel *)
  ; recmodel
  ; adam
  ; loss
  ; mnist
  }
;;

let batches = Base.(10 ** 2)

let train t =
  let open Torch in
  for batch_idx = 1 to batches do
    let batch_images, batch_labels =
      Dataset_helper.train_batch t.mnist ~batch_size:t.batch_size ~batch_idx
    in
    let batch_images = Tensor.reshape ~shape:[ t.batch_size; 1; 28; 28 ] batch_images in
    if t.with_reconstruction
    then (
      let output, probs =
        CapsNetWithReconstruction.forward t.recmodel batch_images batch_labels
      in
      let rec_loss = Tensor.mse_loss output batch_labels in
      let margin_loss = MarginLoss.forward t.loss probs batch_labels true in
      let loss = Tensor.add (Tensor.mul_scalar rec_loss (Scalar.f 0.0005)) margin_loss in
      Optimizer.backward_step t.adam ~loss)
    else ();
    if Base.(batch_idx % 50) = 0
    then (
      let batch_images, batch_labels = t.mnist.test_images, t.mnist.test_labels in
      let batch_images =
        Tensor.slice ~dim:0 ~start:None ~end_:(Some t.batch_size) ~step:1 batch_images
      in
      let batch_labels =
        Tensor.slice ~dim:0 ~start:None ~end_:(Some t.batch_size) ~step:1 batch_labels
      in
      let batch_images =
        Tensor.reshape
          ~shape:[ Tensor.shape batch_images |> List.hd; 1; 28; 28 ]
          batch_images
      in
      let output, _probs =
        CapsNetWithReconstruction.forward t.recmodel batch_images batch_labels
      in
      let batch_accuracy =
        Tensor.(argmax ~dim:(-1) output = batch_labels)
        |> Tensor.to_kind ~kind:(T Float)
        |> Tensor.sum
        |> Tensor.float_value
      in
      Stdio.printf "Test acc:%f\n" batch_accuracy);
    Caml.Gc.full_major ()
  done;
  ()
;;
