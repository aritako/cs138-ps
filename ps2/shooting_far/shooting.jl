function f(x, y::Vector)
    y1_prime = y[2]
    y2_prime = y[2] + 3 * y[1] + 8
    return [y1_prime, y2_prime]
end

function rk4(f, x0, y0::Vector, h, iters)
	res = [(x0, y0)]
	for _ in 1:iters
		x, y = res[end]
		k1 = f(x, y)
        k2 = f(x + h / 2, y + (h/2) * k1)
        k3 = f(x + h / 2, y + (h/2) * k2)
        k4 = f(x + h, y + h * k3)

		x_new = x + h
		y_new = y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

		push!(res, (x_new, y_new))
	end
	return res
end

function shooting(f, x0, y0::Vector, h, iters)
	shooting_figure = Figure()
	shooting_axis = Axis(
		shooting_figure[1,1],
		title = "Shooting Method with y'_0 = $(y0[2])",
		xlabel = "x",
		ylabel = "y",
	)
	points = rk4(f, x0, y0, h, iters)
	x_values = [p[1] for p in points]
	y_values = [p[2][1] for p in points]
	println(points)
	scatter!(shooting_axis, x_values, y_values, colormap = :dense)
	return shooting_figure
end