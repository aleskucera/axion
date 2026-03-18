"""
Curling box trajectory optimization using Dojo (Julia differentiable physics).

Comparable to examples/comparison/curling_box/curling_box_axion.py.

Optimizes the initial Y-velocity of a box sliding on a frictional ground plane
(mu=0.15). Sustained ground contact throughout the episode.
Gradients flow through BPTT via get_maximal_gradients!.

Box maximal state: [x(3), v(3), q(4), ω(3)] = 13 elements
Attjac state:      [x(3), v(3), φ(3), ω(3)] = 12 elements
adj[5] = ∂L/∂vy after the full backward pass.

Usage:
    ~/.juliaup/bin/julia +1.10 --startup-file=no examples/comparison/curling_box/curling_box_dojo.jl
"""

using Dojo
using LinearAlgebra
using Printf

# ======================== Constants ========================

const DT           = 3e-2
const T_STEPS      = 66        # ~2.0 s total
const BOX_HALF     = 0.2       # half-extent (box is 0.4×0.4×0.4 m)
const BOX_MASS     = 6.4       # density 100 × (0.4)³ = 6.4 kg
const BOX_INIT_Z   = 0.21      # box center height at t=0

const INIT_VEL_Y   = 1.0      # initial guess
const TARGET_VEL_Y = 2.5      # target to recover

const LEARNING_RATE = 0.01
const MAX_GRAD      = 20.0

# ======================== Build mechanism ========================

function build_box(; timestep=DT, gravity=-9.81)
    origin = Origin{Float64}()
    box    = Box(BOX_HALF, BOX_HALF, BOX_HALF, BOX_MASS; name=:box)  # half-extents
    j      = JointConstraint(Floating(origin, box; spring=0.0, damper=0.0), name=:base_joint)

    # 4 bottom corners of the box (body frame, z pointing up)
    # Bottom face is at z = -BOX_HALF in body frame
    corners = [
        [-BOX_HALF, -BOX_HALF, -BOX_HALF],
        [-BOX_HALF,  BOX_HALF, -BOX_HALF],
        [ BOX_HALF, -BOX_HALF, -BOX_HALF],
        [ BOX_HALF,  BOX_HALF, -BOX_HALF],
    ]
    contacts = [
        contact_constraint(box, [0.0, 0.0, 1.0];
            friction_coefficient = 0.15,
            contact_origin       = c,
            contact_radius       = 0.0,
            contact_type         = :nonlinear,
            name                 = Symbol("corner_$(i)"))
        for (i, c) in enumerate(corners)
    ]

    return Mechanism(origin, [box], [j], contacts; gravity, timestep)
end

# ======================== State utilities ========================

# Maximal state layout: [x(3), v(3), q(4), ω(3)] = 13 elements
function initial_maximal_state(vy::Float64)
    z = zeros(13)
    z[1:3]   = [0.0, 0.0, BOX_INIT_Z]  # position
    z[4:6]   = [0.0, vy, 0.0]           # linear velocity (Y only)
    z[7:10]  = [1.0, 0.0, 0.0, 0.0]    # quaternion (identity)
    z[11:13] = zeros(3)                 # angular velocity
    return z
end

box_xyz(z::Vector) = z[1:3]

# ======================== Rollout ========================

function rollout_with_gradients(mech::Mechanism, z0::Vector{Float64})
    nu = input_dimension(mech)  # 6 for Floating joint, all zero
    u  = zeros(nu)

    zs  = Vector{Vector{Float64}}(undef, T_STEPS + 1)
    Jzs = Vector{Matrix{Float64}}(undef, T_STEPS)

    zs[1] = z0
    set_maximal_state!(mech, z0)

    z = copy(z0)
    for t in 1:T_STEPS
        Jz, _ = get_maximal_gradients!(mech, z, u)
        z       = get_maximal_state(mech)
        zs[t+1] = z
        Jzs[t]  = Jz
    end

    return zs, Jzs
end

# ======================== Loss and BPTT gradient ========================

"""
Compute trajectory loss and gradient w.r.t. initial Y-velocity via BPTT.

Loss = Σ_t ||xyz_t - xyz_target_t||²

Attjac ordering for a free box: [x(3), v(3), φ(3), ω(3)] → adj[5] = ∂L/∂vy.
"""
function loss_and_grad(zs, Jzs, target_xyz)
    L   = 0.0
    adj = zeros(12)  # attjac: 12 elements

    for t in T_STEPS:-1:1
        delta     = box_xyz(zs[t+1]) - target_xyz[t+1]
        L        += dot(delta, delta)
        adj[1:3] .+= 2.0 .* delta
        adj       = Jzs[t]' * adj
    end

    # adj is now ∂L/∂z0 in attjac space; Y-velocity is at index 5
    grad_vy = adj[5]
    return L, grad_vy
end

# ======================== Main ========================

function main()
    save_path = nothing
    for i in 1:length(ARGS)-1
        if ARGS[i] == "--save"
            save_path = ARGS[i+1]
        end
    end

    mech = build_box()

    println("Curling box mechanism:")
    println("  Bodies:    ", length(mech.bodies))
    println("  Joints:    ", length(mech.joints))
    println("  Contacts:  ", length(mech.contacts))
    println("  Input dim: ", input_dimension(mech), "  (6 base forces/torques, all zero)")
    println("  T=$T_STEPS steps, dt=$DT s")

    # --- Target episode ---
    println("\nSimulating target episode (vy=$TARGET_VEL_Y)...")
    z0_target   = initial_maximal_state(TARGET_VEL_Y)
    target_zs, _ = rollout_with_gradients(mech, z0_target)
    target_xyz   = [box_xyz(target_zs[t]) for t in 1:T_STEPS+1]
    p = target_xyz[end]
    @printf("  Final XYZ: (%.3f, %.3f, %.3f)\n", p[1], p[2], p[3])

    # --- Optimization ---
    vy = INIT_VEL_Y
    println("\nOptimizing: lr=$LEARNING_RATE (gradient descent, max_grad=$MAX_GRAD)")

    iters_log = Int[]
    loss_log  = Float64[]
    time_log  = Float64[]

    for i in 1:30
        t0 = time()
        z0 = initial_maximal_state(vy)
        zs, Jzs = rollout_with_gradients(mech, z0)

        L, grad_vy = loss_and_grad(zs, Jzs, target_xyz)
        grad_clamped = clamp(grad_vy, -MAX_GRAD, MAX_GRAD)
        vy = vy - LEARNING_RATE * grad_clamped

        t_iter = (time() - t0) * 1000
        @printf("Iter %3d: loss=%.4f | vy=%.4f | grad=%.4f | t=%.0fms\n",
            i, L, vy, grad_vy, t_iter)
        push!(iters_log, i - 1)
        push!(loss_log,  L)
        push!(time_log,  t_iter)

        # L < 1e-4 && (println("Converged!"); break)
    end

    if save_path !== nothing
        mkpath(dirname(abspath(save_path)))
        open(save_path, "w") do io
            println(io, "{")
            println(io, "  \"simulator\": \"Dojo\",")
            println(io, "  \"problem\": \"curling_box\",")
            println(io, "  \"dt\": $DT,")
            println(io, "  \"T\": $T_STEPS,")
            println(io, "  \"iterations\": $(iters_log),")
            println(io, "  \"loss\": $(loss_log),")
            println(io, "  \"time_ms\": $(time_log)")
            println(io, "}")
        end
        println("Saved to $save_path")
        return
    end

    println("\nOptimized vy: $vy  (target: $TARGET_VEL_Y)")
end

main()
