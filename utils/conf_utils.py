def get_open_modules(switch_config):
    open_list = []
    for module in switch_config.keys():
        if switch_config[module]:
            open_list.append(module)
    return open_list
