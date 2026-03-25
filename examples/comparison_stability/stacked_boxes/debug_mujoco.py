"""Debug MuJoCo stacked-boxes: check static equilibrium and height trajectory."""
import string
import mujoco
import numpy as np

DENSITY = 1500.0
HX1 = 0.2; HX2 = 0.8; HX3 = 1.6
Z1 = HX1; Z2 = 2*HX1 + HX2; Z3 = 2*HX1 + 2*HX2 + HX3

_XML_TEMPLATE = string.Template("""
<mujoco model="stacked_boxes">
  <option gravity="0 0 -9.81" timestep="$dt" iterations="20" ls_iterations="20"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1"
          friction="0.1 0.1 0.01" solref="$solref" solimp="0.99 0.999 0.001"/>
    <body name="box1" pos="0 0 $z1">
      <freejoint/>
      <inertial mass="$m1" pos="0 0 0" diaginertia="$i1 $i1 $i1"/>
      <geom type="box" size="$hx1 $hx1 $hx1"
            friction="0.1 0.1 0.01" solref="$solref" solimp="0.99 0.999 0.001"/>
    </body>
    <body name="box2" pos="0 0 $z2">
      <freejoint/>
      <inertial mass="$m2" pos="0 0 0" diaginertia="$i2 $i2 $i2"/>
      <geom type="box" size="$hx2 $hx2 $hx2"
            friction="0.1 0.1 0.01" solref="$solref" solimp="0.99 0.999 0.001"/>
    </body>
    <body name="box3" pos="0 0 $z3">
      <freejoint/>
      <inertial mass="$m3" pos="0 0 0" diaginertia="$i3 $i3 $i3"/>
      <geom type="box" size="$hx3 $hx3 $hx3"
            friction="0.1 0.1 0.01" solref="$solref" solimp="0.99 0.999 0.001"/>
    </body>
  </worldbody>
</mujoco>
""")

def _mass(hx): return DENSITY * (2*hx)**3
def _inertia(hx): return _mass(hx) * (2*hx)**2 / 6.0

def run_debug(dt, solref, n_steps=200):
    xml = _XML_TEMPLATE.substitute(
        dt=dt, solref=solref,
        z1=Z1, z2=Z2, z3=Z3,
        hx1=HX1, hx2=HX2, hx3=HX3,
        m1=_mass(HX1), m2=_mass(HX2), m3=_mass(HX3),
        i1=_inertia(HX1), i2=_inertia(HX2), i3=_inertia(HX3),
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"\n--- dt={dt:.5f}s, solref={solref}, {n_steps} steps ---")
    print(f"  Initial box3 z = {data.qpos[16]:.6f}  (Z3={Z3:.4f})")

    for step in range(n_steps):
        mujoco.mj_step(model, data)
        h = data.qpos[16]
        if not np.isfinite(h):
            print(f"  Step {step+1}: NaN/Inf! UNSTABLE")
            return
        dev = abs(h - Z3)
        if step < 10 or step % 20 == 0 or dev > 0.05:
            print(f"  Step {step+1:4d}  t={data.time:.4f}s  box3_z={h:.6f}  dev={dev:.6f}")
        if dev > 1.0:
            print(f"  >> Deviation >1m at step {step+1}, stopping early")
            return

    final_h = data.qpos[16]
    print(f"  Final box3_z={final_h:.6f}  dev={abs(final_h - Z3):.6f} m")

# Test multiple dt values with multiple solref settings
for solref in ["0.001 1", "0.002 1", "0.005 1"]:
    for dt in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
        run_debug(dt, solref, n_steps=100)
