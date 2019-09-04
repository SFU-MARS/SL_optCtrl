import os,sys
import matlab.engine

import numpy as np
if __name__ == "__main__":
    mateng = matlab.engine.start_matlab()
    track_data = mateng.load('/home/xlv/Desktop/SL_optCtrl/MPC/LTV MPC Racing Reference Tracking with Boundary/track2.mat', nargout=1)
    track_data = track_data['track2']
    track_inner = np.array(track_data['inner'])
    track_outer = np.array(track_data['outer'])

    track_inner = np.transpose(track_inner)
    track_outer = np.transpose(track_outer)

    inner_num = np.shape(track_inner)[0]
    outer_num = np.shape(track_outer)[0]
    with open("track.world", 'w') as worldfile:
        worldfile.write("<model name = 'track'>\n")
        worldfile.write("<pose frame = ''> 0 0 0 0 -0 0 </pose>\n")
        worldfile.write("<scale>1 1 1</scale>\n")

        for i in range(inner_num):
            worldfile.write("<link name='inner_%d'>\n" %i)
            worldfile.write("<pose frame=''>%f %f %f 0 -0 0</pose>\n" %(track_inner[i][0], track_inner[i][1],0))
            worldfile.write("<velocity>0 0 0 0 -0 0</velocity>\n")
            worldfile.write("<acceleration>0 0 0 0 -0 0</acceleration>\n")
            worldfile.write("<wrench>0 0 0 0 -0 0</wrench>\n")
            worldfile.write("</link>\n")
        for j in range(outer_num):
            worldfile.write("<link name='outer_%d'>\n" %j)
            worldfile.write("<pose frame=''>%f %f %f 0 -0 0</pose>\n" %(track_outer[j][0], track_outer[j][1],0))
            worldfile.write("<velocity>0 0 0 0 -0 0</velocity>\n")
            worldfile.write("<acceleration>0 0 0 0 -0 0</acceleration>\n")
            worldfile.write("<wrench>0 0 0 0 -0 0</wrench>\n")
            worldfile.write("</link>\n")
        worldfile.write("</model>\n")

        worldfile.write("<!--##################################################### -->\n")

        worldfile.write("<model name='track'>\n")
        worldfile.write("<pose frame=''> 0 0 0 0 -0 0 </pose>\n")
        for i in range(inner_num):
            worldfile.write("<link name='inner_%d'\n>" %i)
            worldfile.write("<pose frame=''>%f %f %f 0 -0 0</pose>\n" %(track_inner[i][0], track_inner[i][1],0))
            worldfile.write("<collision name='collision'>\n")
            worldfile.write("<geometry>\n")
            worldfile.write("<box>\n")
            worldfile.write("<size>0.1 0.1 0.7</size>\n")
            worldfile.write("</box>\n")
            worldfile.write("</geometry>\n")
            worldfile.write("<max_contacts>10</max_contacts>\n")
            worldfile.write("<surface>\n")
            worldfile.write("<contact>\n")
            worldfile.write("<ode/>\n")
            worldfile.write("</contact>\n")
            worldfile.write("<bounce/>\n")
            worldfile.write("<friction>\n")
            worldfile.write("<torsional>\n")
            worldfile.write("<ode/>\n")
            worldfile.write("</torsional>\n")
            worldfile.write("<ode/>\n")
            worldfile.write("</friction>\n")
            worldfile.write("</surface>\n")
            worldfile.write("</collision>\n")
            worldfile.write("<visual name='visual'>\n")
            worldfile.write("<geometry>\n")
            worldfile.write("<box>\n")
            worldfile.write("<size>0.1 0.1 0.7</size>\n")
            worldfile.write("</box>\n")
            worldfile.write("</geometry>\n")
            worldfile.write("<material>\n")
            worldfile.write("<script>\n")
            worldfile.write("<uri>file://media/materials/scripts/gazebo.material</uri>\n")
            worldfile.write("<name>Gazebo/Wood</name>\n")
            worldfile.write("</script>\n")
            worldfile.write("</material>\n")
            worldfile.write("</visual>\n")
            worldfile.write("<self_collide>0</self_collide>\n")
            worldfile.write("<enable_wind>0</enable_wind>\n")
            worldfile.write("<kinematic>0</kinematic>\n")
            worldfile.write("</link>\n")
        for j in range(outer_num):
            worldfile.write("<link name='outer_%d'\n>" % j)
            worldfile.write("<pose frame=''>%f %f %f 0 -0 0</pose>\n" % (track_outer[j][0], track_outer[j][1], 0))
            worldfile.write("<collision name='collision'>\n")
            worldfile.write("<geometry>\n")
            worldfile.write("<box>\n")
            worldfile.write("<size>0.1 0.1 0.7</size>\n")
            worldfile.write("</box>\n")
            worldfile.write("</geometry>\n")
            worldfile.write("<max_contacts>10</max_contacts>\n")
            worldfile.write("<surface>\n")
            worldfile.write("<contact>\n")
            worldfile.write("<ode/>\n")
            worldfile.write("</contact>\n")
            worldfile.write("<bounce/>\n")
            worldfile.write("<friction>\n")
            worldfile.write("<torsional>\n")
            worldfile.write("<ode/>\n")
            worldfile.write("</torsional>\n")
            worldfile.write("<ode/>\n")
            worldfile.write("</friction>\n")
            worldfile.write("</surface>\n")
            worldfile.write("</collision>\n")
            worldfile.write("<visual name='visual'>\n")
            worldfile.write("<geometry>\n")
            worldfile.write("<box>\n")
            worldfile.write("<size>0.1 0.1 0.7</size>\n")
            worldfile.write("</box>\n")
            worldfile.write("</geometry>\n")
            worldfile.write("<material>\n")
            worldfile.write("<script>\n")
            worldfile.write("<uri>file://media/materials/scripts/gazebo.material</uri>\n")
            worldfile.write("<name>Gazebo/Wood</name>\n")
            worldfile.write("</script>\n")
            worldfile.write("</material>\n")
            worldfile.write("</visual>\n")
            worldfile.write("<self_collide>0</self_collide>\n")
            worldfile.write("<enable_wind>0</enable_wind>\n")
            worldfile.write("<kinematic>0</kinematic>\n")
            worldfile.write("</link>\n")

        worldfile.write("<static> 1 </static>\n")
        worldfile.write("</model>")








