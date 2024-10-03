```julia

function fsub_iip(du, u, p, t)
    e = 2.7
    du[1] = (1+u[2]-sin(t))*u[4] + cos(t)
    du[2] = cos(t)
    du[3] = u[4]
    du[4] = (u[1]-sin(t))*(u[4]-e^t)
end

function gsub_iip(res, u, p, t)
    res[1] = u[1] # t==0.0
    res[2] = u[3] - 1 # t == 0.0
    res[3] = u[2] - sin(1.0) # t == 1.0
end

fun_iip = BVPFunction(fsub_iip, gsub_iip, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
prob_iip = BVProblem(fun_iip, [0.0, 0.0, 0.0, 0.0], (0.0, 1.0))
sol = solve(prob_iip, Ascher4(zeta = [0.0, 0.0, 1.0]), dt = 0.2)
```
#=
function fsub_iip(du, u, p, t)
    du[1] = -u[3]
    du[2] = -u[3]
    du[3] = u[2] - sin(t-1)
end

function gsub_iip(res, u, p, t)
    res[1] = u[1] # t==1.0
    res[2] = u[2] # t == 1.0
end
=#
#=testing oop=#

#=
function fsub_oop(u, p, t)
    e = 2.7
    [(1+u[2]-sin(t))*u[4] + cos(t),
    cos(t),
    u[4],
    (u[1]-sin(t))*(u[4]-e^t)]
end

function gsub_oop(u, p, t)
    return [u[1], # t==0.0
    u[3] - 1, # t == 0.0
    u[2] - sin(1.0)] # t == 1.0
end
=#


#=
fun_iip = BVPFunction(f, bc, mass_matrix = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0])
#fun_oop = BVPFunction(fsub_oop, gsub_oop, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])

prob_iip = BVProblem(fun_iip, [0.0, 0.0, 0.0, 0.0, 0.0], (0.0, 1.0), zeta = [0.0, 1.0, 1.0])
#prob_oop = BVProblem(fun_oop, [0.0, 0.0, 0.0, 0.0], (0.0, 1.0), zeta = [0.0, 0.0, 1.0])
dt = 0.1
test(prob_iip, Ascher4(), dt)
=#
#fun_iip = BVPFunction(fsub_iip, gsub_iip, mass_matrix = [1 0 0; 0 1 0; 0 0 0])
#fun_oop = BVPFunction(fsub_oop, gsub_oop, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
#prob_iip = BVProblem(fun_iip, [0.0, 0.0, 0.0], (0.0, 1.0), zeta = [1.0, 1.0])
#prob_oop = BVProblem(fun_oop, [0.0, 0.0, 0.0, 0.0], (0.0, 1.0), zeta = [0.0, 0.0, 1.0])
```