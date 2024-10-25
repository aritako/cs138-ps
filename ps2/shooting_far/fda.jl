function fda_M(m::Int)
	h = 1/m
	
	# Create an m x m zero matrix
	M = zeros(m, m)
	M[1,1] = M[m,m] = 1

	# Fill the matrix for interior points
	for i in 2:m-1
		M[i, i-1] = 2 + h
		M[i, i] = -4 - 6 * h^2
		M[i, i+1] = 2 - h
	end

	return M
end

function fda_b(m::Int)
	h = 1/m
	b = zeros(m)

	# Fill the vector for interior points
	for i in 2:m-1
		b[i] = 16 * h^2
	end

	b[1] = 1
	b[m] = 10

	return b
end

function fda(m::Int)
	M = fda_M(m)
	b = fda_b(m)

	x = 0:1/(m-1):1
	y = M \ b

	fig = Figure()
	ax = Axis(fig[1, 1],
		title="Finite Difference Approximation Trajectory",
		xlabel="Distance (x)",
		ylabel="Height (y)")

	scatter!(ax, x, y, color=:green)
	return fig
end