def restart_carla_docker():
    """Restart the CARLA Docker container"""
    import subprocess
    import time

    try:
        # Stop existing CARLA container
        subprocess.run(['docker', 'stop', 'carla'], check=False)
        subprocess.run(['docker', 'rm', 'carla'], check=False)

        # Start new CARLA container with audio disabled
        subprocess.run(
            [
                'docker',
                'run',
                '-d',
                '--name=carla',
                '--privileged',
                '--gpus',
                'all',
                '--net=host',
                '-v',
                '/tmp/.X11-unix:/tmp/.X11-unix:rw',
                '-e',
                'SDL_AUDIODRIVER=dummy',  # Add this line
                '-e',
                'ALSA_CARD=Dummy',  # Add this line
                'carlasim/carla:0.9.15',
                '/bin/bash',
                './CarlaUE4.sh',
                '-RenderOffScreen',
                '-nosound'  # Add -nosound flag
            ],
            check=True)

        # Wait for CARLA to initialize
        time.sleep(10)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restarting CARLA container: {e}")
        return False
