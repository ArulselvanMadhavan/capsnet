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

val make
  :  ?batch_size:int
  -> ?test_batch_size:int
  -> ?epochs:int
  -> ?lr:float
  -> ?no_cuda:bool
  -> ?seed:int
  -> ?log_interval:int
  -> ?routing_iterations:int
  -> ?with_reconstruction:bool
  -> unit
  -> t
