"""
    eventtimes([rng], rate, t_end::Real) -> Vector{Float64}

Event times of a Poisson process of constant `rate` over `[0, t_end]`, in
increasing order.

The clock starts at the first event, so `out[1]` is exactly zero. That discards
the waiting time before the first event, which is the price of having the
acquisition begin when something happens rather than at an arbitrary instant.
"""
function eventtimes(rng::AbstractRNG, rate::Real, t_end::Real)
    rate > 0 || throw(ArgumentError("rate must be positive, got $rate"))
    t_end >= 0 || throw(ArgumentError("t_end must be non-negative, got $t_end"))

    # Sample in unit-rate time and rescale once at the end, so the loop is a
    # single randexp and an add.
    horizon = t_end * rate
    out = Float64[]
    sizehint!(out, ceil(Int, horizon))
    t = 0.0
    while true
        t += randexp(rng)
        t >= horizon && break
        push!(out, t)
    end
    isempty(out) && return out

    # Shift to the first event *before* rescaling — both are in unit-rate time.
    t0 = out[1]
    out .= (out .- t0) ./ rate
    return out
end

"""
    eventtimes([rng], rate, n::Integer) -> Vector{Float64}

The first `n` event times of a Poisson process of constant `rate`, starting at
zero. Fixes the *count* and lets the duration float, the complement of the
`t_end` method and what you want when comparing fits at a matched number of
events.

`n` events span `n-1` exponential gaps, so the duration is `Gamma(n-1, 1/rate)`.
"""
function eventtimes(rng::AbstractRNG, rate::Real, n::Integer)
    rate > 0 || throw(ArgumentError("rate must be positive, got $rate"))
    n >= 0 || throw(ArgumentError("n must be non-negative, got $n"))

    out = Vector{Float64}(undef, n)
    n == 0 && return out
    out[1] = 0.0
    n == 1 && return out
    cumsum!(view(out, 2:n), randexp(rng, n - 1) ./ rate)
    return out
end

eventtimes(rate::Real, horizon) = eventtimes(default_rng(), rate, horizon)
