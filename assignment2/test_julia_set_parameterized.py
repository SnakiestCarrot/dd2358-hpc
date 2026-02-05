
import julia_set as js
import pytest

@pytest.mark.parametrize("desired_width, max_iterations, expected_sum", [
    (1000, 300, 33219980),
    (500, 200, 5798200),
])
def test_calculate_z_serial_purepython_better(desired_width, max_iterations, expected_sum):
    x_step = (js.x2 - js.x1) / desired_width
    y_step = (js.y1 - js.y2) / desired_width

    x = []
    y = []
    ycoord = js.y2
    while ycoord > js.y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = js.x1
    while xcoord < js.x2:
        x.append(xcoord)
        xcoord += x_step

    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(js.c_real, js.c_imag))

    output = js.calculate_z_serial_purepython(max_iterations, zs, cs)

    assert sum(output) == expected_sum
