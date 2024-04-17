import mujoco
import time
import matplotlib.pyplot as plt
import numpy as np

# exit(0)
DEG2RAD = np.pi/180


def sim_display_forces():
    # visualize contact frames and forces, make body transparent
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # tweak scales of contact visualization elements
    model.vis.scale.contactwidth = 0.1
    model.vis.scale.contactheight = 0.03
    model.vis.scale.forcewidth = 0.05
    model.vis.map.force = 0.3
    return options


def simulate(model, data):

    options = sim_display_forces()

    # random initial rotational velocity:
    mujoco.mj_resetData(model, data)
    data.qvel[3:6] = 5 * np.random.randn(3)

    # create axes
    ax1 = plt.subplot(111)  # create axes
    im1 = ax1.imshow(renderer.render())  # create image plot
    plt.ion()  # interactive mode

    ##############################

    print('Total number of DoFs in the model:', model.nv)
    print('Generalized positions:', data.qpos)
    print('Generalized velocities:', data.qvel)

    while True: # viewer.is_running():
        step_start = time.time()
        # Step the simulation.
        mujoco.mj_step(model, data)

        renderer.update_scene(data, "track", options)
        im1.set_data(renderer.render())
        plt.pause(0.02)

        ###########################################

        # print('Total number of DoFs in the model:', model.nv)
        # print('Generalized positions:', data.qpos)
        # print('Generalized velocities:', data.qvel)
        # print()

    plt.ioff()  # due to infinite loop, this gets never called.
    plt.show()

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("red_cube/red_cube.xml")
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, 400, 600)
    mujoco.mj_forward(model, data)

    simulate(model, data)