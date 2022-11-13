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
  let no_cuda = not no_cuda && not (Torch.Cuda.is_available ()) in
  Torch_core.Wrapper.manual_seed seed;
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
