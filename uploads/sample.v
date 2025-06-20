module example_circuit(input a, b, c, d, e, output x, y, z, w);
  wire w1, w2, w3, w4, w5, w6;
  
  // First level gates
  and g1(w1, a, b);
  or g2(w2, b, c);
  not g3(w3, c);
  xor g4(w4, d, e);
  
  // Second level gates
  nand g5(w5, w1, w2);
  nor g6(w6, w3, w4);
  
  // Output gates
  and g7(x, w5, w6);
  or g8(y, w1, w4);
  buf g9(z, w2);
  not g10(w, w6);
endmodule
