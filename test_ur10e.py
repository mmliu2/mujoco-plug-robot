import mujoco
import time
import matplotlib.pyplot as plt
import numpy as np

# exit(0)
DEG2RAD = np.pi/180
NUM_JOINTS = 6

def get_angles(data):
    return data.qpos

def set_angles(data, desired_angles):
    data.qpos = desired_angles

def get_pos(data):
    return data.xpos

def set_pos(data, desired_pos):
    data.xpos = desired_pos

def get_velocities(data):
    return data.qvel

def set_velocities(data, desired_velocities):
    data.qvel = desired_velocities
    # data.actuator_velocity = desired_velocities


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


def display(floats_list, name=""):
    print("    " + name + ":\t", end ='')
    print("[", end ='')
    strings = [f" {num:.6f}" if num >= 0 else f"{num:.6f}" for num in floats_list]
    print("  ".join(strings), end="")
    print("]", end ='\n')

def simulate(model, data):

    options = sim_display_forces()

    # create axes
    ax1 = plt.subplot(111)  # create axes
    im1 = ax1.imshow(renderer.render())  # create image plot
    plt.ion()  # interactive mode

    ##############################

    # set_velocities(data, [100] * NUM_JOINTS)
    print('Total number of DoFs in the model:', model.nv)
    print('Positions:', data.qpos)
    print('Velocities:', data.qvel)

    desired_angles = [0, 0, -90*DEG2RAD, 0, 0, 0]
    set_angles(data, desired_angles)

    ##############################

    t0 = time.time()
    iteration = 0
    while True:
        # Step the simulation.
        mujoco.mj_step(model, data)
        renderer.update_scene(data, "cam1", options)
        im1.set_data(renderer.render())
        plt.pause(0.02)

        ###########################################

        if iteration%100 == 0:
            vels = [0] * NUM_JOINTS
            vels[2] = 20
            # vels[0] = 10  # for scene2
            set_velocities(data, vels)
            print("Set velocity")

        if iteration%100 == 90:
            desired_angles = [0, 0, -90 * DEG2RAD, 0, 0, 0]
            set_angles(data, desired_angles)
            vels = [0] * NUM_JOINTS
            set_velocities(data, vels)
            print("Reset")

        if iteration%10 == 0:
            # display(get_angles(data), "qpos")
            # display(get_pos(data)[-1], "xpos")
            display(get_velocities(data), "qvel")
            # print()

        iteration += 1


if __name__ == "__main__":

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur10e_wall/scene_no_solimp_solref.xml")
    # model = mujoco.MjModel.from_xml_path("universal_robots_ur10e_wall/scene.xml")
    data = mujoco.MjData(model)

    # Simulation
    renderer = mujoco.Renderer(model, 400, 640)
    mujoco.mj_forward(model, data)
    simulate(model, data)
