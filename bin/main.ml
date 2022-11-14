let () =
  let c = Capsnet.make () in
  Capsnet.train c;
  print_endline "Hello, World!"
;;
