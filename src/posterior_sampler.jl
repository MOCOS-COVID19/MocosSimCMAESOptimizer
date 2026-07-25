struct SurrogatePosterior
    center::Vector{Float64}
    scale::Vector{Float64}
    linear::Vector{Float64}
    curvature::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    temperature::Float64
end

function read_jsonl(path::String)
    rows = Any[]
    isfile(path) || error("Archive not found: $path")
    for line in eachline(path)
        isempty(strip(line)) && continue
        push!(rows, JSON.parse(line))
    end
    isempty(rows) && error("Archive is empty: $path")
    return rows
end

function archive_bounds(rows, dim::Int)
    lower = fill(-Inf, dim)
    upper = fill(Inf, dim)
    for row in rows
        x = Float64.(row["x_evaluated"])
        for i in 1:min(dim, length(x))
            lower[i] = min(lower[i], x[i])
            upper[i] = max(upper[i], x[i])
        end
    end
    return lower, upper
end

function archive_objective(row)
    if haskey(row, "vector_log_likelihood") && isfinite(Float64(row["vector_log_likelihood"]))
        return -Float64(row["vector_log_likelihood"])
    end
    return Float64(row["score"])
end

function fit_diagonal_surrogate(rows, state_path::String; temperature::Float64=1.0)
    state = JSON.parsefile(state_path)
    center = Float64.(state["mean"])
    dim = length(center)
    sigma = Float64(state["sigma"])
    covariance = Matrix{Float64}(undef, dim, dim)
    for i in 1:dim
        covariance[i, :] .= Float64.(state["covariance"][i])
    end
    eig = eigen(Symmetric(covariance + 1e-10I))
    eigvals = max.(eig.values, 1e-10)
    transform = sigma .* eig.vectors * Diagonal(sqrt.(eigvals))
    scale = max.(sqrt.(diag(transform * transform')), 1e-6)

    scores = Float64[archive_objective(row) for row in rows if isfinite(archive_objective(row))]
    isempty(scores) && error("Archive contains no finite scores")
    score_scale = max(median(abs.(scores .- minimum(scores))), 1e-6)
    ordered = sort(rows, by=archive_objective)
    keep = ordered[1:min(length(ordered), max(20, 4 * dim))]
    lower, upper = archive_bounds(keep, dim)
    best = Float64.(keep[1]["x_evaluated"])

    linear = zeros(dim)
    curvature = ones(dim)
    for i in 1:dim
        xs = Float64[]
        ys = Float64[]
        ws = Float64[]
        for row in keep
            x = Float64.(row["x_evaluated"])
            length(x) < i && continue
            dx = (x[i] - best[i]) / scale[i]
            push!(xs, dx)
            push!(ys, (archive_objective(row) - minimum(scores)) / score_scale)
            push!(ws, exp(-0.5 * (length(xs) - 1) / max(length(keep), 1)))
        end
        length(xs) < 3 && continue
        design = hcat(ones(length(xs)), xs, 0.5 .* xs .^ 2)
        weights = Diagonal(ws)
        ridge = 1e-3 .* Matrix{Float64}(I, 3, 3)
        beta = (design' * weights * design + ridge) \ (design' * weights * ys)
        linear[i] = beta[2] / scale[i]
        curvature[i] = max(beta[3] / scale[i]^2, 1e-3)
    end
    return SurrogatePosterior(best, scale, linear, curvature, lower, upper, temperature)
end

logistic(x::Float64) = 1.0 / (1.0 + exp(-clamp(x, -40.0, 40.0)))

function unconstrained_to_x(q::Vector{Float64}, posterior::SurrogatePosterior)
    x = similar(q)
    jacobian = 0.0
    dx_dq = similar(q)
    for i in eachindex(q)
        range = posterior.upper[i] - posterior.lower[i]
        if !isfinite(range) || range <= 0.0
            x[i] = posterior.center[i] + posterior.scale[i] * q[i]
            dx_dq[i] = posterior.scale[i]
        else
            s = logistic(q[i])
            x[i] = posterior.lower[i] + range * s
            dx_dq[i] = range * s * (1.0 - s)
            jacobian += log(max(dx_dq[i], 1e-12))
        end
    end
    return x, dx_dq, jacobian
end

function surrogate_logdensity_gradient(x::Vector{Float64}, posterior::SurrogatePosterior)
    z = (x .- posterior.center) ./ posterior.scale
    potential = posterior.temperature * sum(
        posterior.linear .* z .+ 0.5 .* posterior.curvature .* z .^ 2
    )
    gradient = posterior.temperature .* (
        posterior.linear .+ posterior.curvature .* z
    ) ./ posterior.scale
    return -potential, gradient
end

function transformed_logdensity_gradient(q::Vector{Float64}, posterior::SurrogatePosterior)
    x, dx_dq, jacobian = unconstrained_to_x(q, posterior)
    logp_x, grad_x = surrogate_logdensity_gradient(x, posterior)
    grad_q = grad_x .* dx_dq
    for i in eachindex(q)
        if isfinite(posterior.upper[i] - posterior.lower[i])
            s = logistic(q[i])
            grad_q[i] += 1.0 - 2.0 * s
        end
    end
    return logp_x + jacobian, grad_q, x
end

function nuts_leapfrog(q, p, grad, step_size, direction, posterior)
    p_new = p .+ direction .* (0.5 * step_size) .* grad
    q_new = q .+ direction .* step_size .* p_new
    logp_new, grad_new, x_new = transformed_logdensity_gradient(q_new, posterior)
    p_new = p_new .+ direction .* (0.5 * step_size) .* grad_new
    return q_new, p_new, grad_new, logp_new, x_new
end

function nuts_is_uturn(q_minus, q_plus, p_minus, p_plus)
    delta = q_plus .- q_minus
    return dot(delta, p_minus) < 0.0 || dot(delta, p_plus) < 0.0
end

function nuts_sample(rng, q0::Vector{Float64}, posterior::SurrogatePosterior;
                     draws::Int=1000, warmup::Int=500, max_depth::Int=8,
                     step_size::Float64=0.05, target_accept::Float64=0.8)
    q = copy(q0)
    dim = length(q)
    samples = Vector{Vector{Float64}}()
    diagnostics = Any[]
    log_step = log(step_size)
    running_accept = target_accept
    total = warmup + draws
    for iteration in 1:total
        logp, grad, x = transformed_logdensity_gradient(q, posterior)
        p0 = randn(rng, dim)
        h0 = -logp + 0.5 * dot(p0, p0)
        log_slice = log(rand(rng)) - h0
        q_minus = copy(q)
        q_plus = copy(q)
        p_minus = copy(p0)
        p_plus = copy(p0)
        grad_minus = copy(grad)
        grad_plus = copy(grad)
        q_proposal = copy(q)
        logp_proposal = logp
        n_valid = 1
        accepted_sum = 0.0
        proposal_count = 0
        for depth in 0:max_depth
            direction = rand(rng, Bool) ? 1.0 : -1.0
            steps = 2^depth
            for _ in 1:steps
                if direction > 0
                    q_plus, p_plus, grad_plus, logp_new, x_new = nuts_leapfrog(q_plus, p_plus, grad_plus, exp(log_step), direction, posterior)
                else
                    q_minus, p_minus, grad_minus, logp_new, x_new = nuts_leapfrog(q_minus, p_minus, grad_minus, exp(log_step), direction, posterior)
                end
                h_new = -logp_new + 0.5 * dot(direction > 0 ? p_plus : p_minus, direction > 0 ? p_plus : p_minus)
                if isfinite(h_new)
                    proposal_count += 1
                    accepted_sum += exp(min(0.0, h0 - h_new))
                    if log_slice <= -h_new
                        n_valid += 1
                        if rand(rng) < 1.0 / n_valid
                            q_proposal = direction > 0 ? copy(q_plus) : copy(q_minus)
                            logp_proposal = logp_new
                        end
                    end
                end
                abs(h_new - h0) > 1000.0 && break
                nuts_is_uturn(q_minus, q_plus, p_minus, p_plus) && break
            end
            nuts_is_uturn(q_minus, q_plus, p_minus, p_plus) && break
        end
        q = q_proposal
        if iteration <= warmup
            acceptance = accepted_sum / max(proposal_count, 1)
            log_step += 0.02 * (acceptance - target_accept)
            running_accept = 0.95 * running_accept + 0.05 * acceptance
        else
            _, _, x = transformed_logdensity_gradient(q, posterior)
            push!(samples, x)
        end
        push!(diagnostics, Dict(
            "iteration" => iteration,
            "warmup" => iteration <= warmup,
            "step_size" => exp(log_step),
            "accepted_steps" => accepted_sum,
            "valid_states" => n_valid,
            "running_acceptance" => running_accept,
        ))
    end
    return samples, diagnostics
end

function run_nuts_from_archive(archive_path::String;
                               state_path::String=joinpath(dirname(archive_path), "cma_sampling_state.json"),
                               output_path::String=joinpath(dirname(archive_path), "posterior_samples.json"),
                               draws::Int=1000, warmup::Int=500, max_depth::Int=8,
                               step_size::Float64=0.05, temperature::Float64=1.0,
                               seed::Int=42)
    rows = read_jsonl(archive_path)
    state = JSON.parsefile(state_path)
    posterior = fit_diagonal_surrogate(rows, state_path; temperature=temperature)
    q0 = zeros(length(posterior.center))
    for i in eachindex(q0)
        if isfinite(posterior.upper[i] - posterior.lower[i])
            fraction = clamp(
                (posterior.center[i] - posterior.lower[i]) /
                max(posterior.upper[i] - posterior.lower[i], 1e-12),
                1e-6, 1.0 - 1e-6,
            )
            q0[i] = log(fraction / (1.0 - fraction))
        end
    end
    samples, diagnostics = nuts_sample(
        MersenneTwister(seed), q0, posterior;
        draws=draws, warmup=warmup, max_depth=max_depth, step_size=step_size,
    )
    posterior_mean = isempty(samples) ? posterior.center : vec(mean(reduce(hcat, samples), dims=2))
    posterior_covariance = if length(samples) > 1
        sample_matrix = reduce(hcat, samples)
        centered = sample_matrix .- posterior_mean
        centered * centered' / (size(sample_matrix, 2) - 1)
    else
        Diagonal(posterior.scale .^ 2) |> Matrix
    end
    param_names = haskey(state, "param_names") ? String.(state["param_names"]) : ["x[$i]" for i in eachindex(posterior.center)]
    safe_save_json(output_path, Dict(
        "sampler" => "NUTS",
        "surrogate" => "diagonal_quadratic_from_cma_archive",
        "archive_path" => archive_path,
        "state_path" => state_path,
        "draws" => draws,
        "warmup" => warmup,
        "max_depth" => max_depth,
        "step_size" => step_size,
        "temperature" => temperature,
        "seed" => seed,
        "center" => posterior.center,
        "scale" => posterior.scale,
        "linear" => posterior.linear,
        "curvature" => posterior.curvature,
        "param_names" => param_names,
        "posterior_mean" => posterior_mean,
        "posterior_covariance" => posterior_covariance,
        "samples" => samples,
        "diagnostics" => diagnostics,
    ); label="posterior_samples")
    return output_path
end

function aggregate_stage_archive(stage_root::String)
    paths = String[]
    for entry in sort(readdir(stage_root))
        startswith(entry, "iter_") || continue
        path = joinpath(stage_root, entry, "posterior_training_data.jsonl")
        isfile(path) && push!(paths, path)
    end
    isempty(paths) && return nothing
    aggregate_path = joinpath(stage_root, "posterior_training_data_all.jsonl")
    open(aggregate_path, "w") do out
        for path in paths
            for line in eachline(path)
                println(out, line)
            end
        end
    end
    return aggregate_path
end

function run_nuts_from_stage(stage_root::String;
                             output_path::String=joinpath(stage_root, "posterior_samples.json"),
                             draws::Int=500, warmup::Int=250, max_depth::Int=8,
                             step_size::Float64=0.05, temperature::Float64=1.0,
                             seed::Int=42)
    archive_path = aggregate_stage_archive(stage_root)
    archive_path === nothing && return nothing
    state_iterations = [
        parse(Int, replace(entry, "iter_" => ""))
        for entry in readdir(stage_root)
        if startswith(entry, "iter_") &&
           isfile(joinpath(stage_root, entry, "cma_sampling_state.json"))
    ]
    isempty(state_iterations) && error("No CMA sampling state found in $stage_root")
    state_path = joinpath(stage_root, "iter_$(maximum(state_iterations))", "cma_sampling_state.json")
    return run_nuts_from_archive(
        archive_path;
        state_path=state_path,
        output_path=output_path,
        draws=draws,
        warmup=warmup,
        max_depth=max_depth,
        step_size=step_size,
        temperature=temperature,
        seed=seed,
    )
end

function posterior_reusable_state(posterior_path::String)
    posterior = JSON.parsefile(posterior_path)
    mean_vector = Float64.(posterior["posterior_mean"])
    covariance = Matrix{Float64}(undef, length(mean_vector), length(mean_vector))
    for i in eachindex(mean_vector)
        covariance[i, :] .= Float64.(posterior["posterior_covariance"][i])
    end
    covariance = 0.5 .* (covariance + covariance')
    stds = sqrt.(max.(diag(covariance), 1e-8))
    sigma = clamp(median(stds), 0.02, 0.5)
    normalized_covariance = covariance ./ sigma^2
    normalized_covariance += 1e-6I
    return Dict(
        "stage" => "posterior",
        "param_names" => posterior["param_names"],
        "param_ranges" => [Any[] for _ in mean_vector],
        "mean" => mean_vector,
        "sigma" => sigma,
        "covariance" => normalized_covariance,
        "source" => posterior_path,
    )
end
