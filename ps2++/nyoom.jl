# Basic nyoom
function nyoom(h, q)
	heights = []
	for i in 1:q
		push!(heights, h+i)
	end
	return heights
end

# nyoom with velocity
function nyoom(h, v, q)
	heights = []
	for i in 1:q
		push!(heights,h + v*i)
	end
	return heights
end

# multidimensional nyoom 
function nyoom(h::Vector, v::Vector, q)
	heights = []
	for i in 1:length(h)
		temp = []
		for j in 1:q
			push!(temp, h[i] + v[i] * j)
		end
		push!(heights, temp)
	end
	return heights
end