include("device_tests.jl")
test_device_jacobians(CPU(), Tsit5())
