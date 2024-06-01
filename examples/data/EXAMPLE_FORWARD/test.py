import yaml

# Load the YAML file
with open('../adjoint_source_model.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Modify the record_length value
data['time_axis']['record_length'] = 7200.  # Change this to your desired value


# Save the modified YAML file
with open('inparam.source.yaml', 'w') as file:
    yaml.safe_dump(data, file, default_flow_style=False)