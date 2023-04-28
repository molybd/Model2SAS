import torch

from model2sas import *


def test_normal_flow():
    part1 = MathPart(filename=r'resources/exp_models/cylinder_z.py', device='cpu')
    part1.set_params(R=10, H=30, sld_value=1)
    part1.sampling()
    part1.scatter()

    part2 = StlPart(filename=r'resources/exp_models/torus.stl', device='cpu', sld_value=2)
    part2.sampling()
    part2.scatter()

    plot_model(part1, part2)

    assembly = Assembly(part1, part2)
    plot_model(assembly)

    q = torch.linspace(0.01, 2, steps=200)
    I = assembly.measure(q)
    plot_1d_sas(q, [I, 10*I], name='cylinder', mode=['markers', 'lines'], title='test')

    det1 = Detector((981, 1043), 172e-6)
    det1.set_sdd(1.5)
    det1.translate(0, -50e-3)
    det2 = Detector((514, 1030), 75e-6)
    det2.set_sdd(0.2)
    det2.pitch(torch.pi/6)
    det2.translate(0, 60e-3)

    wavelength = 1.342
    qcoord1 = det1.get_reciprocal_coord(wavelength)
    qcoord2 = det2.get_reciprocal_coord(wavelength)
    I2d1 = assembly.get_sas(*qcoord1)
    I2d2 = assembly.get_sas(*qcoord2)
    plot_2d_sas(I2d1, savename='temp/test_2dhtml.html', show=True, title='2d plot')
    plot_surface((*qcoord1, I2d1), (*qcoord2, I2d2))


def test_transform():
    part1 = MathPart(filename=r'resources/exp_models/cylinder_z.py', device='cpu')
    part1.set_params(R=10, H=20, sld_value=1)
    part1.sampling()
    part1.rotate((1,0,0), torch.pi/5.4)
    part1.translate(120, 34, -103)
    part1.scatter()

    part2 = StlPart(filename=r'resources/exp_models/torus.stl', device='cpu', sld_value=2)
    part2.rotate((1,1,0), torch.pi/3.1)
    part2.translate(131, 37, -97)
    part2.sampling()
    part2.scatter()

    plot_model(part1, part2, type='voxel')

    assembly = Assembly(part1, part2)
    plot_model(assembly)


def test_mathmodel_generate():
    mathcls = gen_math_model_class(
        name = 'hollow_sphere',
        params = dict(r=5, t=10, sld_value_scale=1),
        coord = 'sph',
        bound_point=('r+t', 'r+t', 'r+t'),
        shape_description = '(u>=r) & (u<=(r+t))',
        sld_description = 'sld_value_scale * (u * torch.cos(v) * torch.sin(w))',
    )
    part1 = MathPart(math_model_class=mathcls)
    part1.set_params(r=30, t=20, sld_value_scale=3)
    part1.sampling()
    plot_model(part1)
    plot_model(part1, type='volume')

    part1.scatter()
    q = torch.linspace(0.01, 1, steps=200)
    I = part1.measure(q)
    plot_1d_sas(q, I)

    filename = 'resources/exp_models/savedmodel.py'
    with open(filename, 'w') as f:
        f.write(
            gen_math_model_class_sourcecode(*tuple(part1.math_model.info.values()))
        )
    part2 = MathPart(filename=filename)
    part2.set_params(r=30, t=20, sld_value_scale=3)
    part2.sampling()
    plot_model(part2, type='volume')


def test_detector_simulation():
    det1 = Detector((981, 1043), 172e-6)
    det1.set_sdd(1.5)
    det1.translate(0, -50e-3)
    det2 = Detector((514, 1030), 75e-6)
    det2.set_sdd(0.2)
    det2.pitch(torch.pi/6)
    det2.translate(0, 60e-3)
    
    plot_real_detector(
        (det1.x, det1.y, det1.z), (det2.x, det2.y, det2.z)
    )

    part1 = MathPart(filename=r'resources/exp_models/cylinder_z.py', device='cpu')
    part1.set_params(R=10, H=30, sld_value=1)
    part1.sampling()
    part1.scatter()
    part2 = StlPart(filename=r'resources/exp_models/torus.stl', device='cpu', sld_value=2)
    part2.sampling()
    part2.scatter()
    assembly = Assembly(part1, part2)

    wavelength = 1.342
    qcoord1 = det1.get_reciprocal_coord(wavelength)
    qcoord2 = det2.get_reciprocal_coord(wavelength)
    I1 = assembly.get_sas(*qcoord1)
    I2 = assembly.get_sas(*qcoord2)
    plot_real_detector(
        (det1.x, det1.y, det1.z, I1), (det2.x, det2.y, det2.z, I2)
    )

