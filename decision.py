import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, velocity is at max and absolute
                # steering angle is at max, the rover often gets stuck in a loop
                if np.fabs(Rover.vel) >= Rover.max_vel and np.fabs(Rover.steer) == 15.0: 
                    Rover.throttle = 0
                    # Rover.brake = Rover.brake_set
                    Rover.steer = 0.5*Rover.steer

                # If mode is forward, velocity is 0 m/s, throttle is at max
                #steering angle is at max. Rover gets stuck and cannot get out
                elif Rover.vel == 0.0 and Rover.throttle == Rover.throttle_set and\
                np.fabs(Rover.steer) == 15.0:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = -Rover.steer
                    Rover.mode = 'stop'

                # If mode is forward, velocity is 0 m/s, throttle is at max
                # steering angle is at max. Rover is spinning at one place
                elif Rover.vel == 0.0 and Rover.throttle == 0.0 and\
                np.fabs(Rover.steer) == 15.0:
                    Rover.steer = 0
                    Rover.throttle = Rover.throttle_set

                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle \
                elif Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                    Rover.brake = 0
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                else: # Else coast
                    Rover.throttle = 0
                    Rover.brake = 0
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                # Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set*np.random.uniform(0,1,100)
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set

    return Rover