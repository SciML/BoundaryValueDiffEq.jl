for order in (4, 6)
    alg = Symbol("MIRKN$(order)")
    f = Symbol("constructMIRKN$(order)")
    @eval constructMIRKN(::$(alg), ::Type{T}) where {T} = $(f)(T)
end

function constructMIRKN4(::Type{T}) where {T}
    # RK coefficients tableau
    s = 3
    c = [0, 1, 1 // 2, 3 // 10]
    v = [0, 1, 1 // 2, 3 // 10]
    w = [0, 0, -3 // 20, -1]
    b = [1 // 6, 0, 1 // 3, 0]
    x = [0 0 0 0
         0 0 0 0
         1//80 1//80 0 0
         4//25 87//500 561//1000 0]
    vp = [0, 1, 1 // 2, 2 // 5]
    xp = [0 0 0 0
          0 0 0 0
          1//8 -1//8 0 0
          349//3000 -281//3000 -46//375 0]
    bp = [1 // 6, 1 // 6, 2 // 3, 0]

    TU = MIRKNTableau(s, T.(c), T.(v), T.(w), T.(b), T.(x), T.(vp), T.(bp), T.(xp))
    return TU
end

function constructMIRKN6(::Type{T}) where {T}
    # RK coefficients tableau
    s = 5
    c = [0, 1, 1 // 5, 4 // 5, 1 // 2, 1 // 2, 2 // 5, 4 // 25]
    v = [0, 1, 1 // 10, 9 // 10, 1 // 2, 9 // 25, 4 // 5, 2 // 5]
    w = [0, 0, -1 // 50, -3 // 25, -1 // 5, -1 // 4, 0, 9 // 25]
    b = [1 // 16, 0, 25 // 108, 25 // 432, 4 // 27, 0]
    x = [0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0
         -7//1500 -2//375 0 0 0 0 0 0
         -2//375 -7//1500 0 0 0 0 0 0
         2329//61440 2329//61440 -5//12288 -5//12288 0 0 0 0
         911//38400 23//1536 1073//13824 737//13824 137//5400 0 0 0
         -317//12500 -3//3125 -428//3375 -581//13500 0 -10448//84375 0 0
         0 -7701259//1562500 -38497579//93750000 -745411//46875000 0 -7158056//9765625 10337749//15625000 0]
    vp = [0, 1, 13 // 125, 112 // 125, 1 // 2, 12 // 25,
        -17497 // 12500, -44493991 // 31250000]
    xp = [0 0 0 0 0 0 0 0
          0 0 0 0 0 0 0 0
          16//125 -4//125 0 0 0 0 0 0
          4//125 -16//125 0 0 0 0 0 0
          -13//256 13//256 75//256 -75//256 0 0 0 0
          183//6400 -167//6400 1165//6912 -1085//6912 4//675 0 0 0
          29817//200000 17817//200000 223//320 127//320 14//25 -288//3125 0 0
          76329639//500000000 45632679//500000000 11534329//20000000 7996921//20000000 1//2 56448//1953125 -12936//78125 0]
    bp = [1 // 16, 1 // 16, 125 // 432, 125 // 432, 8 // 27, 0, 0, 0]

    TU = MIRKNTableau(s, T.(c), T.(v), T.(w), T.(b), T.(x), T.(vp), T.(bp), T.(xp))
    return TU
end
