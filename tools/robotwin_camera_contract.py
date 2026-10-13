"""Select native 640x480 RoboTwin cameras without changing their field of view."""


def configure_native_cameras(evaluator, config):
    for key in ('head_camera_type', 'wrist_camera_type'):
        current = config['camera'][key]
        original = evaluator.get_camera_config(current)
        if (original['w'], original['h']) == (640, 480):
            continue
        candidate = f'Large_{current}'
        target = evaluator.get_camera_config(candidate)
        if (target['w'], target['h'], target['fovy']) != (640, 480, original['fovy']):
            raise ValueError(f'No matching native 640x480 camera for {current}')
        config['camera'][key] = candidate
    config['head_camera_w'], config['head_camera_h'] = 640, 480
    return config


def install_camera_contract(evaluator):
    original = evaluator.load_task_args

    def load_task_args(args):
        config, embodiment = original(args)
        return configure_native_cameras(evaluator, config), embodiment

    evaluator.load_task_args = load_task_args
