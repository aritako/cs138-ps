function uniform(p, a, b) :: Vector
	return a:(b-a)/(p-1):b
end

function chebyshev(p, a, b) :: Vector
	x = zeros(p)
	
	for i in 1:p
		xi = cos((2i - 1) * π / (2p))
		x[i] = (a + b)/2 + (b - a)/2 * xi
	end
	
	return reverse(x)
end

function monomial(x::Vector, j)
	return x .^ (j-1)
end

function gaussian(x::Vector, j)
	return exp.(-1.9044 * (x[j] .- x) .^ 2)
end

function Lmonomial(x::Vector, j)
	(j-2) * (j-1) * x .^ (j-3) - (j-1) * x .^ (j-2) .- 3 * monomial(x, j)
end

function Lgaussian(x::Vector, j)
	ep_sqr = 3.8088
	return ep_sqr * gaussian(x, j) .* (ep_sqr * (x[j] .- x).^2 .- 1) - ep_sqr * (x[j] .- x) .* gaussian(x, j) .- 3 * gaussian(x, j)
end

function colloc_M_b(
	p,
	xa, ya,
	xb, yb,
	point_fn, basis_fn
)
	# Generate collocation points
	x = point_fn(p, xa, xb)

	# Initialize matrix M and vector b 
	M = zeros(p, p)
	b = zeros(p)

	
	# Fill in the matrix M and vector b
	for j in 1:p
		phi_points = basis_fn(x, j)
		
		Lphi_points = basis_fn == gaussian ? 
		Lgaussian(x, j) : Lmonomial(x, j)
		
		M[1, 	 j] = phi_points[1]
		M[2:p-1, j] = Lphi_points[2:p-1]
		M[p, 	 j] = phi_points[p]
		
	end
	
	b = ones(p) * 8
	b[1] = ya
	b[end] = yb
	
	return (M, b)
end

function colloc(
	p,
	xa, ya,
	xb, yb,
	point_fn, basis_fn
)
	
	M, b = colloc_M_b(p, xa, ya, xb, yb, point_fn, basis_fn)
	w = M \ b

	x = point_fn(p, xa, xb)
	y = w[1] * basis_fn(x, 1)
	for i in 2:p
		y = y + w[i] * basis_fn(x, i)
	end

	fig = Figure()
	ax = Axis(fig[1, 1], 
		title="Collocation Trajectory (Uniform Points, Monomial Basis)",
		# title="Collocation Trajectory (Uniform Points, Gaussian Basis)",
		# title="Collocation Trajectory (Chebyshev Points, Gaussian Basis)",
		# title="Collocation Trajectory (Chebyshev Points, Monomial Basis)",
		xlabel="Distance (x)", 
		ylabel="Height (y)")

	scatter!(ax, x, y, color=:blue)
	return fig
end