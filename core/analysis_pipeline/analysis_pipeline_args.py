from conf.service_args import service_config


task_candidate_list = list(service_config["ModuleSwitch"].keys())
task_candidate_list.remove("analysis_pipeline")


if __name__ == "__main__":
    print(task_candidate_list)
