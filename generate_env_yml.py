def generate_environment_yml(requirements_file, environment_file):
    with open(requirements_file, 'r') as req_file:
        packages = [line.strip() for line in req_file.readlines() if line.strip()]

    with open(environment_file, 'w') as env_file:
        env_file.write('name: axisem3d_output\n')
        env_file.write('channels:\n')
        env_file.write('  - conda-forge\n')
        env_file.write('dependencies:\n')
        for package in packages:
            env_file.write(f'  - {package}\n')

    print(f'Successfully generated {environment_file}')


if __name__ == '__main__':
    requirements_file = 'requirements.txt'
    environment_file = 'environment.yml'
    generate_environment_yml(requirements_file, environment_file)
