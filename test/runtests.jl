using SimSMLM
using Test
using StableRNGs
using StaticArrays
using Statistics
using LinearAlgebra
using Distributions
using Random: randexp

# StableRNGs rather than a seeded default RNG: Julia does not guarantee its default random stream across minor versions, so statistical assertions written against it would break on upgrade for no good reason.
rng() = StableRNG(20260905)

@testset "SimSMLM.jl" begin
    @testset "time-varying parameters" begin
        @test at(3.0, 1.0) == 3.0
        @test at(t -> 2t, 4.0) == 8.0
        @test at(SA[1.0, 2.0], 9.0) == SA[1.0, 2.0]
        @test at(I, 0.0) === I
    end

    @testset "deformations" begin
        u = SA[1.0, 0.0, 0.0]
        @test place(Static{3}(), u, 5.0) == u

        m = Similarity{3}(scale=t -> 100 + 20t, shift=t -> SA[10t, 0.0, 0.0])
        @test place(m, u, 0.0) ≈ SA[100.0, 0.0, 0.0]
        @test place(m, u, 1.0) ≈ SA[130.0, 0.0, 0.0]   # 120 from growth + 10 drift

        r = Similarity{2}(rot=t -> rotation(t))
        @test place(r, SA[1.0, 0.0], π / 2) ≈ SA[0.0, 1.0] atol = 1e-12

        R = rotation(SA[0.0, 0.0, 1.0], π / 2)
        @test R * SA[1.0, 0.0, 0.0] ≈ SA[0.0, 1.0, 0.0] atol = 1e-12
        @test det(R) ≈ 1.0

        # rotation(n): the shortest rotation taking +z onto n. The reference is LocMoFit's rotmat(θ, ϕ), the fitters' convention, with n at polar θ, azimuth ϕ.
        rotmat(θ, ϕ) = [
            1-(1-cos(θ))*cos(ϕ)^2 -(1-cos(θ))*sin(ϕ)*cos(ϕ) sin(θ)*cos(ϕ);
            -(1-cos(θ))*sin(ϕ)*cos(ϕ) 1-(1-cos(θ))*sin(ϕ)^2 sin(θ)*sin(ϕ);
            -sin(θ)*cos(ϕ) -sin(θ)*sin(ϕ) cos(θ)
        ]
        dir(θ, ϕ) = SA[sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
        for (θ, ϕ) in ((0.3, 1.1), (π / 2, -2.0), (2.9, 0.4), (π - 1e-9, 2.5))
            Rn = rotation(dir(θ, ϕ))
            @test Rn ≈ rotmat(θ, ϕ) atol = 1e-12
            @test Rn * SA[0.0, 0.0, 1.0] ≈ dir(θ, ϕ)
            @test Rn' * Rn ≈ I atol = 1e-12
            @test det(Rn) ≈ 1
        end
        @test rotation(SA[0.0, 3.0, 4.0]) ≈ rotation(SA[0.0, 0.6, 0.8])   # need not be normalized
        @test rotation(SA[0.0, 0.0, 2.0]) == SMatrix{3,3,Float64}(I)
        @test rotation(SA[0.0, 0.0, -1.0]) ≈ rotmat(π, 0.0) atol = 1e-12
        @inferred rotation(SA[1.0, 2.0, 3.0])
        @test_throws ArgumentError rotation(SA[0.0, 0.0, 0.0])
        @test_throws DimensionMismatch rotation([1.0, 0.0])

        inner = Similarity{3}(scale=2.0)
        outer = Drift(SA[1.0, 0.0, 0.0])
        @test place(outer ∘ inner, u, 0.0) ≈ SA[3.0, 0.0, 0.0]
        @test place(Drift{3}(t -> SA[t, 0.0, 0.0]), u, 4.0) ≈ SA[5.0, 0.0, 0.0]
    end

    @testset "bend: constant-area cap" begin
        u = SA[0.6, 0.0, 0.8]
        # κ = ±1 is the unit sphere itself, moved so its north pole is the apex
        @test place(Bend(1.0), u, 0.0) ≈ u - SA[0.0, 0.0, 1.0]
        @test place(Bend(-1.0), u, 0.0) ≈ SA[0.6, 0.0, 0.2]
        # κ = 0 is a flat disk, and the map is continuous through it
        @test place(Bend(0.0), u, 0.0)[3] == 0
        @test place(Bend(1e-12), u, 0.0) ≈ place(Bend(0.0), u, 0.0) atol = 1e-12
        # the north pole is the apex; the south pole lands on the rim, ρ′ = 2√(1-κ²)
        @test place(Bend(0.3), SA[0.0, 0.0, 1.0], 0.0) == zero(SVector{3,Float64})
        @test norm(place(Bend(0.3), SA[0.0, 0.0, -1.0], 0.0)[SA[1, 2]]) ≈
            2 * sqrt(1 - 0.3^2)
        # a label keeps its depth fraction h/2 as the cap bends
        @test place(Bend(0.2), u, 0.0)[3] / 0.2 ≈ place(Bend(0.8), u, 0.0)[3] / 0.8
        @test place(Bend(t -> 0.1t), u, 5.0) == place(Bend(0.5), u, 5.0)
        @inferred place(Bend(t -> 0.1t), u, 2.0)

        # Uniform material points land uniformly on a cap of area 4π. For κ ≠ 0 every point lies on the sphere of radius 1/|κ| through the apex, and by the hat-box theorem uniform area means a uniform depth h = -z/κ on [0, 2]; for κ = 0 it means ρ′² uniform on [0, 4].
        r = rng()
        n = 40_000
        sites = sample_sites(r, SphereSurface{3}(), n)
        for κ in (0.4, -0.9)
            xs = [place(Bend(κ), s, 0.0) for s in sites]
            c = SA[0.0, 0.0, -1 / κ]
            @test all(x -> norm(x - c) ≈ 1 / abs(κ), xs)
            h = [-x[3] / κ for x in xs]
            @test mean(h) ≈ 1 atol = 4 * (2 / sqrt(12)) / sqrt(n)
            @test count(<(0.5), h) / n ≈ 1 / 4 atol = 4 * sqrt(3 / 16 / n)
            @test maximum(h) ≈ 2 atol = 0.01
        end
        disk = [place(Bend(0.0), s, 0.0) for s in sites]
        ρ² = [x[1]^2 + x[2]^2 for x in disk]
        @test all(x -> x[3] == 0, disk)
        @test mean(ρ²) ≈ 2 atol = 4 * (4 / sqrt(12)) / sqrt(n)
        @test maximum(ρ²) <= 4 + 1e-12

        # rotation(n) orients a cap by its outward normal at the apex: every point lies on the sphere of radius s/κ centred s/κ behind the apex
        n, s, κ, apex = normalize(SA[1.0, -2.0, 0.5]), 30.0, 0.6, SA[5.0, 0.0, -3.0]
        m = Similarity{3}(scale=s, rot=rotation(n), shift=apex) ∘ Bend(κ)
        centre = apex - (s / κ) * n
        @test all(x -> norm(place(m, x, 0.0) - centre) ≈ s / κ, sites)
        @test place(m, SA[0.0, 0.0, 1.0], 0.0) ≈ apex

        @test_throws ArgumentError Bend(1.5)
        @test_throws DomainError place(Bend(t -> 2.0), u, 0.0)
    end

    @testset "shapes" begin
        r = rng()
        @test sample_site(r, SphereSurface{3}()) isa SVector{3,Float64}

        s = sample_sites(r, SphereSurface{3}(), 2000)
        @test length(s) == 2000
        @test all(x -> norm(x) ≈ 1.0, s)
        @test mean(s) ≈ zero(SVector{3,Float64}) atol = 0.08   # isotropic

        b = sample_sites(r, Ball{3}(), 2000)
        @test all(x -> norm(x) <= 1.0 + 1e-12, b)
        # uniform in the ball => E[r] = D/(D+1)
        @test mean(norm, b) ≈ 3 / 4 rtol = 0.03

        ring = sample_sites(r, Ring{3}(), 2000)
        @test all(x -> x[3] == 0.0, ring)
        @test all(x -> norm(x) ≈ 1.0, ring)

        # in the plane the unit circle *is* the unit sphere, so Ring{2} and SphereSurface{2} are the same distribution; they diverge from D = 3 up
        ang(v) = atan(v[2], v[1])
        bins(v) =
            [count(x -> lo <= ang(x) < lo + π / 4, v) for lo in range(-π, 3π / 4, length=8)]
        @test all(≈(5000; rtol=0.08), bins(sample_sites(r, Ring{2}(), 40_000)))
        @test all(≈(5000; rtol=0.08), bins(sample_sites(r, SphereSurface{2}(), 40_000)))

        seg = sample_sites(r, LineSegment(SA[0.0, 0.0], SA[1.0, 1.0]), 100)
        @test all(x -> x[1] ≈ x[2], seg)

        M = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test tosites(M) == [SA[1.0, 4.0], SA[2.0, 5.0], SA[3.0, 6.0]]
        # a point cloud now resamples with replacement, so points recur
        pc = PointCloud(M)
        draws = sample_sites(r, pc, 3000)
        @test all(in(tosites(M)), draws)
        @test length(unique(draws)) == 3
        @test count(==(SA[1.0, 4.0]), draws) / 3000 ≈ 1 / 3 atol = 0.03
    end

    @testset "event times" begin
        r = rng()
        λ, T = 50.0, 100.0

        t = eventtimes(r, λ, T)
        @test issorted(t)
        @test all(x -> 0.0 <= x <= T, t)
        @test t[1] == 0.0                     # clock starts at the first event
        @test length(t) ≈ λ * T rtol = 0.05
        gaps = diff(t)
        @test mean(gaps) ≈ 1 / λ rtol = 0.05
        @test std(gaps) ≈ 1 / λ rtol = 0.05   # exponential: mean == std

        # a window too short for anything to happen must come back empty, not reach for out[1]
        @test eventtimes(r, 1.0, 0.0) == Float64[]

        @test_throws ArgumentError eventtimes(r, 0.0, 1.0)
        @test_throws ArgumentError eventtimes(r, -1.0, 1.0)
        @test_throws ArgumentError eventtimes(r, 1.0, -1.0)
    end

    @testset "photon count is geometric" begin
        r = rng()
        h, τ = 500.0, 0.1
        o = Optics(uncertainty=ShotNoise{3}(100.0), brightness=h, ontime=τ)
        @test meanphotons(o) == h * τ
        @test photondist(o) == Geometric(1 / (1 + h * τ))

        # the derivation's claim: marginalising N|T ~ Poisson(hT), T ~ Exp(τ) gives Geometric(1/(1+hτ)) exactly, so one draw replaces two
        twostep = [rand(r, Poisson(h * randexp(r) * τ)) for _ in 1:200_000]
        onestep = rand(r, photondist(o), 200_000)
        @test mean(twostep) ≈ h * τ rtol = 0.02
        @test mean(onestep) ≈ h * τ rtol = 0.02
        @test var(twostep) ≈ h * τ * (1 + h * τ) rtol = 0.03
        @test var(onestep) ≈ h * τ * (1 + h * τ) rtol = 0.03
        # P(N = 0) = p, the fraction of events that produce nothing at all
        @test count(==(0), onestep) / 200_000 ≈ 1 / (1 + h * τ) rtol = 0.05

        # simulate draws from this law and drops the zeros, so the counts that survive follow P(N=n) = (1-p)^(n-1) p with mean 1 + hτ
        kept = filter(!iszero, onestep)
        @test minimum(kept) >= 1
        @test mean(kept) ≈ 1 + h * τ rtol = 0.02
        @test count(==(1), kept) / length(kept) ≈ 1 / (1 + h * τ) rtol = 0.06

        @test_throws ArgumentError Optics(
            uncertainty=ShotNoise{3}(100.0), brightness=0.0, ontime=τ
        )
    end

    @testset "uncertainty models" begin
        @test psfwidths(3, 130.0) == SA[130.0, 130.0, 325.0]
        @test psfwidths(2, 100.0) == SA[100.0, 100.0]

        ideal = ShotNoise{3}(130.0)
        @test uncertainty(ideal, 100) ≈ SA[13.0, 13.0, 32.5]
        @test uncertainty(ideal, 400) ≈ uncertainty(ideal, 100) ./ 2   # σ ∝ 1/√n

        # Stallinga-Rieger reduces to the shot-noise bound as background and pixel size vanish, and is strictly worse than it otherwise. Note v = 2π sₐ² b / (n a²), so b must go to zero at *fixed* a — shrinking both together holds v at a constant instead.
        sr0 = StallingaRieger{2}(130.0; pixel=1e-3, background=0.0)
        @test uncertainty(sr0, 500) ≈ uncertainty(ShotNoise{2}(130.0), 500) rtol = 1e-5
        sr = StallingaRieger{2}(130.0; pixel=100.0, background=50.0)
        @test all(uncertainty(sr, 500) .> uncertainty(ShotNoise{2}(130.0), 500))

        # the Gaussian mask estimator keeps a 4/3 penalty even with no background
        gm = GaussianMask{2}(130.0; pixel=1e-3, background=0.0)
        @test isapprox(
            uncertainty(gm, 500),
            uncertainty(ShotNoise{2}(130.0), 500) .* (4 / 3);
            rtol=1e-5,
        )

        @test_throws ArgumentError psfwidths(4, 100.0)
    end

    @testset "end-to-end: static sphere" begin
        r = rng()
        sphere = Structure(SphereSurface{3}(), Similarity{3}(scale=100.0))
        locs, truth = simulate(
            r,
            sphere;
            emis_rate=(2000.0),
            t_end=50.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01),
        )

        @test locs isa LocalizationTable{3,Float64}
        @test length(locs) == length(truth)
        @test issorted(locs.t)                       # events arrive in order
        @test all(t -> 0.0 <= t <= 50.0, locs.t)   # clock starts at the first event
        @test all(>=(1), locs.nphotons)

        # λ·T events, of which a fraction p = 1/(1+hτ) emit nothing and are cut
        p0 = 1 / (1 + 1e5 * 0.01)
        @test length(locs) ≈ 2000 * 50 * (1 - p0) rtol = 0.02
        @test all(>=(1), locs.nphotons)          # zeros never reach the table
        @test all(isfinite, precisions(locs))
        @test mean(locs.nphotons) ≈ 1 + 1e5 * 0.01 rtol = 0.02

        # truth lies exactly on the sphere; localizations scatter about it
        @test all(x -> norm(x) ≈ 100.0, truth.position)
        # |x| is biased *outward*: E[|x+ε|] > |x| by Jensen, and the geometric photon count adds a heavy tail of near-dark, very noisy events (the Fréchet tail of σ). The median is the robust summary of the two.
        rad = norm.(locs.position)
        @test median(rad) ≈ 100.0 rtol = 0.02
        @test mean(rad) > median(rad)
        @test count(<(10), locs.nphotons) / length(locs) < 0.02

        # residuals normalised by the reported precision are standard normal
        z = errors(locs, truth) ./ precisions(locs)
        @test mean(z) ≈ 0.0 atol = 0.02
        @test std(vec(z)) ≈ 1.0 rtol = 0.02

        # zero-copy matrix views
        @test size(positions(locs)) == (3, length(locs))
        @test positions(locs) === parent(locs.position)   # the storage itself
        @test positions(locs)[:, 1] == locs[1].position == locs.position[1]
        @test size(truepositions(truth)) == (3, length(truth))

        @test_throws DimensionMismatch errors(locs[1:10], truth)
    end

    @testset "end-to-end: growing, drifting sphere" begin
        sphere = Structure(
            SphereSurface{3}(),
            Similarity{3}(scale=t -> 100 + 20t, shift=t -> SA[10t, 0.0, 0.0]),
        )
        locs, truth = simulate(
            rng(),
            sphere;
            emis_rate=(1000.0),
            t_end=10.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01),
        )
        @test !isempty(locs)

        # the structure really does grow and drift: radius about the instantaneous centre tracks 100 + 20t at every time, not just on average
        for (lo, hi) in ((0.0, 1.0), (4.0, 5.0), (9.0, 10.0))
            idx = findall(t -> lo <= t < hi, locs.t)
            @test !isempty(idx)
            mid = (lo + hi) / 2
            centre = SA[10 * mid, 0.0, 0.0]
            @test isapprox(
                mean(norm(truth.position[i] - centre) for i in idx),
                100 + 20 * mid;
                rtol=0.02,
            )
        end

        # the material coordinate is on the unit sphere and maps back to the true position through the deformation
        i = firstindex(locs)
        @test norm(truth.u[i]) ≈ 1.0
        @test place(truth.motion, truth.u[i], locs.t[i]) ≈ truth.position[i]
    end

    @testset "t_end bounds the acquisition" begin
        sphere = Structure(SphereSurface{3}(), Similarity{3}(scale=100.0))
        kw = (
            emis_rate=5000.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e6, ontime=0.01),
        )
        short, _ = simulate(rng(), sphere; t_end=5.0, kw...)
        long, _ = simulate(rng(), sphere; t_end=10.0, kw...)
        @test length(short) / length(long) ≈ 0.5 rtol = 0.05
        @test all(t -> 0.0 <= t <= 5.0, short.t)

        @test_throws ArgumentError simulate(rng(), sphere; t_end=-1.0, kw...)
    end

    @testset "fixed event count" begin
        r = rng()
        λ = 50.0
        t = eventtimes(r, λ, 10_000)
        @test length(t) == 10_000
        @test issorted(t)
        @test t[1] == 0.0
        gaps = diff(t)
        @test length(gaps) == 9_999            # n events span n-1 gaps
        @test mean(gaps) ≈ 1 / λ rtol = 0.03
        @test std(gaps) ≈ 1 / λ rtol = 0.03
        @test last(t) ≈ 9_999 / λ rtol = 0.03

        # degenerate counts must not reach for a gap that is not there
        @test eventtimes(r, λ, 0) == Float64[]
        @test eventtimes(r, λ, 1) == [0.0]
        @test_throws ArgumentError eventtimes(r, λ, -1)
    end

    @testset "field of view" begin
        f = FieldOfView(SA[-100.0, -50.0], SA[100.0, 150.0])
        @test ndim(f) == 2
        @test volume(f) == 200.0 * 200.0

        r = rng()
        pts = [randin(r, f) for _ in 1:200_000]
        @test all(p -> all(f.lo .<= p .<= f.hi), pts)
        @test mean(pts) ≈ (f.lo .+ f.hi) ./ 2 rtol = 0.02      # uniform => centred
        # uniform on [lo,hi] has variance (hi-lo)^2/12
        @test var([p[1] for p in pts]) ≈ 200.0^2 / 12 rtol = 0.03
        @test var([p[2] for p in pts]) ≈ 200.0^2 / 12 rtol = 0.03

        @test_throws ArgumentError FieldOfView(SA[0.0, 0.0], SA[1.0, 0.0])   # hi == lo
        @test_throws ArgumentError FieldOfView(SA[0.0], SA[1.0, 2.0])        # length mismatch
    end

    @testset "background localizations" begin
        fov = FieldOfView(SA[-400.0, -400.0, -400.0], SA[400.0, 400.0, 400.0])
        sphere = Structure(SphereSurface{3}(), Similarity{3}(scale=100.0))
        λs, λb = 2000.0, 500.0
        μs, μb = 1e5 * 0.01, 40.0        # background is much dimmer than signal
        locs, truth = simulate(
            rng(),
            sphere;
            emis_rate=(λs),
            t_end=20.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01),
            background=Background(rate=λb, fov=fov, meanphotons=μb),
        )

        # the split follows the superposition ratio
        @test nbackground(truth) / length(truth) ≈ λb / (λs + λb) rtol = 0.03
        @test nsignal(truth) + nbackground(truth) == length(truth)
        @test length(locs) == length(truth)

        sigidx = findall(truth.issignal)
        bgidx = findall(!, truth.issignal)

        # signal truth is on the sphere; background truth is uniform in the fov
        @test all(i -> norm(truth.position[i]) ≈ 100.0, sigidx)
        @test all(i -> all(fov.lo .<= truth.position[i] .<= fov.hi), bgidx)
        @test mean(truth.position[i] for i in bgidx) ≈ zero(SVector{3,Float64}) atol = 8.0

        # material coordinates: unit vectors for signal, NaN for background
        @test all(i -> norm(truth.u[i]) ≈ 1.0, sigidx)
        @test all(i -> all(isnan, truth.u[i]), bgidx)

        # both kinds carry real photon counts, and residuals stay standard normal for background too — a spurious event is still localized
        z = errors(locs, truth) ./ precisions(locs)
        @test std(vec(z[:, sigidx])) ≈ 1.0 rtol = 0.03
        @test std(vec(z[:, bgidx])) ≈ 1.0 rtol = 0.04

        # zero background rate is allowed and yields none
        l0, t0 = simulate(
            rng(),
            sphere;
            emis_rate=(λs),
            t_end=2.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01),
            background=Background(rate=0.0, fov=fov, meanphotons=μb),
        )
        @test nbackground(t0) == 0
        @test all(t0.issignal)

        # background has its own photon budget, and being dimmer it localizes worse — both share the PSF, so σ = s/√N is the only difference
        @test mean(locs.nphotons[sigidx]) ≈ 1 + μs rtol = 0.03
        @test mean(locs.nphotons[bgidx]) ≈ 1 + μb rtol = 0.03
        @test median(locs.precision[i][1] for i in bgidx) >
            median(locs.precision[i][1] for i in sigidx)

        @test meanphotons(Background(rate=1.0, fov=fov, meanphotons=7.0)) == 7.0
        @test photondist(Background(rate=1.0, fov=fov, meanphotons=7.0)) == Geometric(1 / 8)

        @test_throws ArgumentError Background(rate=-1.0, fov=fov, meanphotons=μb)
        @test_throws ArgumentError Background(rate=1.0, fov=fov, meanphotons=0.0)
    end

    @testset "end-to-end: bending cap" begin
        # constant area 4π·50², flat at t = 0 and curving to κ = 0.9 at t = 9
        s, apex = 50.0, SA[0.0, 0.0, 100.0]
        cap = Structure(
            SphereSurface{3}(), Similarity{3}(scale=s, shift=apex) ∘ Bend(t -> 0.1t)
        )
        locs, truth = @inferred simulate(
            rng(),
            cap;
            emis_rate=2000.0,
            t_end=9.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01),
        )
        @test !isempty(locs)
        # every true position is the bend of its own material point at its own time: depth below the apex, in units of s, is κ(t)·(1 - u_z)
        @test all(
            i ->
                (apex[3] - truth.position[i][3]) / s ≈
                0.1 * locs.t[i] * (1 - truth.u[i][3]),
            eachindex(truth),
        )
        @test all(
            i -> place(truth.motion, truth.u[i], locs.t[i]) ≈ truth.position[i],
            eachindex(truth),
        )
        z = errors(locs, truth) ./ precisions(locs)
        @test std(vec(z)) ≈ 1.0 rtol = 0.03
    end

    @testset "smaptable" begin
        fov = FieldOfView(SA[-300.0, -300.0, -300.0], SA[300.0, 300.0, 300.0])
        opt = Optics(
            uncertainty=ShotNoise(SA[100.0, 140.0, 300.0]), brightness=1e5, ontime=0.01
        )
        locs, truth = simulate(
            rng(),
            Structure(SphereSurface{3}(), Similarity{3}(scale=100.0));
            emis_rate=500.0,
            t_end=1.0,
            optics=opt,
            background=Background(rate=50.0, fov=fov, meanphotons=40.0),
        )

        tab = @inferred smaptable(locs)
        @test keys(tab) == (
            :x_nm,
            :y_nm,
            :z_nm,
            :x_err_nm,
            :y_err_nm,
            :xy_err_nm,
            :z_err_nm,
            :photons,
            :time_s,
        )
        @test all(c -> length(c) == length(locs), values(tab))
        @test [tab.x_nm tab.y_nm tab.z_nm]' == positions(locs)
        @test [tab.x_err_nm tab.y_err_nm tab.z_err_nm]' == precisions(locs)
        # SMAPpy's lateral precision, the RMS of anisotropic x and y
        @test tab.xy_err_nm ≈ sqrt.((tab.x_err_nm .^ 2 .+ tab.y_err_nm .^ 2) ./ 2)
        @test tab.photons == locs.nphotons
        @test tab.time_s == locs.t
        # copies: editing the table leaves the simulation untouched
        tab.x_nm[1] += 1
        @test positions(locs)[1, 1] != tab.x_nm[1]

        # the truth is a separate table, row-aligned, with the same position names
        gt = @inferred smaptable(truth)
        @test keys(gt) == (:x_nm, :y_nm, :z_nm, :is_signal, :component)
        @test all(c -> length(c) == length(locs), values(gt))
        @test [gt.x_nm gt.y_nm gt.z_nm]' == truepositions(truth)
        @test gt.is_signal == truth.issignal
        @test gt.component == truth.component

        # 2D: no z columns
        l2, t2 = simulate(
            rng(),
            Structure(Ring{2}(), Similarity{2}(scale=50.0));
            emis_rate=500.0,
            t_end=1.0,
            optics=Optics(uncertainty=ShotNoise{2}(100.0), brightness=1e5, ontime=0.01),
        )
        @test keys(smaptable(l2)) ==
            (:x_nm, :y_nm, :x_err_nm, :y_err_nm, :xy_err_nm, :photons, :time_s)
        @test keys(smaptable(t2)) == (:x_nm, :y_nm, :is_signal, :component)
    end

    @testset "composite structures" begin
        ring = Structure(Ring{3}(), Similarity{3}(scale=50.0))
        cap = Structure(
            SphereSurface{3}(),
            Similarity{3}(scale=40.0, shift=SA[0.0, 0.0, 80.0]) ∘ Bend(t -> 0.05t),
        )
        scr = Composite(ring => 1.0, cap => 3.0)
        fov = FieldOfView(SA[-300.0, -300.0, -300.0], SA[300.0, 300.0, 300.0])
        opt = Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01)
        kw = (emis_rate=2000.0, t_end=10.0, optics=opt)
        locs, truth = @inferred simulate(
            rng(), scr; kw..., background=Background(rate=200.0, fov=fov, meanphotons=40.0)
        )

        # each signal event picks component k with probability w_k / Σw
        sig = findall(truth.issignal)
        p = 3 / 4
        @test isapprox(
            count(==(2), truth.component) / length(sig),
            p;
            atol=4 * sqrt(p * (1 - p) / length(sig)),
        )
        # component 0 is exactly the background
        @test (truth.component .== 0) == .!truth.issignal
        @test all(in(0:2), truth.component)

        # each row maps back through its own component's deformation, and the components stay where they belong: the ring in z = 0, the cap within 2·40·κ(t) ≤ 40 nm below its apex at z = 80
        @test truth.motion === (ring.motion, cap.motion)
        @test all(
            i ->
                place(truth.motion[truth.component[i]], truth.u[i], locs.t[i]) ≈
                truth.position[i],
            sig,
        )
        @test all(i -> truth.position[i][3] == 0, findall(==(1), truth.component))
        @test all(i -> 40 <= truth.position[i][3] <= 80, findall(==(2), truth.component))

        # a plain Structure is component 1 throughout, and keeps its own motion
        _, t1 = simulate(rng(), ring; kw...)
        @test all(==(1), t1.component)
        @test t1.motion === ring.motion

        # a zero weight is never drawn
        _, t0 = simulate(rng(), Composite(ring => 0.0, cap => 1.0); kw...)
        @test all(==(2), t0.component)

        a, _ = simulate(rng(), scr; kw...)
        b, _ = simulate(rng(), scr; kw...)
        @test a.position == b.position

        @test_throws ArgumentError Composite()
        @test_throws ArgumentError Composite(ring => -1.0, cap => 1.0)
        @test_throws ArgumentError Composite(ring => 0.0, cap => 0.0)
        @test_throws ArgumentError Composite(ring => Inf, cap => 1.0)
        @test_throws DimensionMismatch Composite(ring => 1.0, Structure(Ring{2}()) => 1.0)
    end

    @testset "nevents and t_end are exclusive" begin
        sphere = Structure(SphereSurface{3}(), Similarity{3}(scale=100.0))
        opt = Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e6, ontime=0.01)
        kw = (emis_rate=(1000.0), optics=opt)

        # nevents counts events; the zero-photon fraction trims the yield
        locs, truth = simulate(rng(), sphere; nevents=20_000, kw...)
        @test length(locs) == length(truth)
        @test length(locs) <= 20_000
        @test length(locs) / 20_000 ≈ 1 - 1 / (1 + 1e6 * 0.01) rtol = 0.01
        @test all(>=(1), locs.nphotons)
        @test all(isfinite, precisions(locs))

        @test_throws ArgumentError simulate(rng(), sphere; kw...)
        @test_throws ArgumentError simulate(rng(), sphere; t_end=1.0, nevents=10, kw...)
    end

    @testset "reproducibility" begin
        s = Structure(SphereSurface{3}(), Similarity{3}(scale=100.0))
        kw = (
            emis_rate=(500.0),
            t_end=2.0,
            optics=Optics(uncertainty=ShotNoise{3}(120.0), brightness=1e5, ontime=0.01),
        )
        a, ta = simulate(rng(), s; kw...)
        b, tb = simulate(rng(), s; kw...)
        @test a.position == b.position
        @test a.nphotons == b.nphotons
        @test ta.position == tb.position

        # the shape/motion convenience form is the same as building a Structure
        c, _ = simulate(rng(), SphereSurface{3}(), Similarity{3}(scale=100.0); kw...)
        @test c.position == a.position
    end
end
