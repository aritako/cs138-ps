# Spline Matrix Generation
function spline_mat(x::Vector, y::Vector)
	n_plus_1 = length(x)
	M = zeros(n_plus_1, n_plus_1)

	M[1, 1] = 1
	M[n_plus_1, n_plus_1] = 1

	for i in 2:n_plus_1-1
		h_i = x[i] - x[i-1]
		h_i_plus_1 = x[i + 1] - x[i]
		u_i = 2 * (h_i + h_i_plus_1)

		M[i, i - 1] = h_i
		M[i, i] = u_i
		M[i, i + 1] = h_i_plus_1
	end
	
	return M
end

# Spline Vector k
function spline_vec(x::Vector, y::Vector)
	n_plus_1 = length(x)
	b = zeros(n_plus_1)

	for i in 2:n_plus_1-1
		h_i = x[i] - x[i-1]
		h_i_plus_1 = x[i + 1] - x[i]

		first_term = (y[i + 1] - y[i])/h_i_plus_1
		second_term = (y[i] - y[i - 1])/h_i
		v_i = 6 * (first_term - second_term)

		b[i] = v_i
	end
	return b
end


# Spline Matrix Solution
function spline_k(M::Matrix, b::Vector)
	return M \ b
end

function gen_spline(x::Vector, y::Vector)
	n_plus_1 = length(x)
	M = spline_mat(x, y)
	b = spline_vec(x, y)
	k = spline_k(M, b)

	splines = []
	
	for i in 1:n_plus_1-1
			h_i = x[i + 1] - x[i]
			
			a = k[i] / (6 * h_i)
			b_coef = k[i + 1] / (6 * h_i)
			c = (y[i + 1] / h_i) - (k[i + 1] * h_i / 6)
			d = (y[i] / h_i) - (k[i] * h_i / 6)

			function p_i(x_val)
					term1 = a * (x[i + 1] - x_val)^3
					term2 = b_coef * (x_val - x[i])^3
					term3 = c * (x_val - x[i])
					term4 = d * (x[i + 1] - x_val)
					return term1 + term2 + term3 + term4
			end

			push!(splines, p_i)
	end

	return splines
end
