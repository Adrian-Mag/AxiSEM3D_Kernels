from ruamel.yaml import YAML

# Create a YAML object
yaml = YAML()

# Load the YAML file
with open('input/inparam.source.yaml', 'r') as file:
    data = yaml.load(file)

# Modify the record_length value
data['time_axis']['record_length'] = 100.  # Change this to your desired value

# Save the modified YAML file
with open('inparam.source.yaml', 'w') as file:
    yaml.dump(data, file)