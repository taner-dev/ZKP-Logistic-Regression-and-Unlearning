pragma circom 2.1.6;

template LogReg(n) {
    signal input x[n];
    signal input w[n];
    signal input b;

    
    signal input y;      // {0,1}
    signal input z;      // angeblicher Score

    signal output out;

    // y muss Bit sein
    y * (y - 1) === 0;

    // z = b + sum(x[i] * w[i])
    signal acc[n + 1];
    acc[0] <== b;
    for (var i = 0; i < n; i++) {
        acc[i + 1] <== acc[i] + x[i] * w[i];
    }

    acc[n] === z;

    
    out <== y;
}

component main = LogReg(3);


