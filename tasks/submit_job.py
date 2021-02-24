from argparse import ArgumentParser
import os

CONFIG_DIR = "../configs/"

if __name__ == '__main__':
    parser = ArgumentParser(prog='PROG')
    parser.add_argument('--project_config_file', '-pf', type=str, required=True)
    parser.add_argument('--project_config_overrides', '-po', type=str)
    parser.add_argument('--model_config_file', '-mf', type=str, required=True)
    parser.add_argument('--model_config_overrides', '-mo', type=str)
    parser.add_argument('--wall_time', '-wt', type=int, default=240)
    parser.add_argument('--gpumem', type=int, default=11000)
    parser.add_argument('--gpuhost', type=str, default='')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--run_name', '-n', type=str, default='')
    parser.add_argument('--best_effort', '-b', action='store_true')
    parser.add_argument('--idempotent', '-i', action='store_true')
    # parse params
    args = parser.parse_args()


    besteffort = "-t besteffort" if args.best_effort else ""
    idempotent = "-t idempotent" if args.idempotent else ""
    resume = "--resume" if args.best_effort and args.idempotent else ""
    if ',' in args.gpuhost:
        hosts = args.gpuhost.split(',')
        host = "and (host=\'" + '\' or host=\''.join(hosts) + '\')'
    else:
        host = f"and (host=\'{args.gpuhost}\')" if args.gpuhost != '' else ''

    if args.gpus > 1 and args.gpuhost != '':
        gpus = f'gpuid={args.gpus},'
    elif args.gpus > 1:
        gpus = f'host=1/gpuid={args.gpus},'
    else:
        gpus = ''

    # SIGINT (2) -> keyboard interrupt signal used by pytorch-lightning for graceful exit
    oar_specifics = f'oarsub {besteffort} {idempotent} --checkpoint 120 --signal 2 ' + \
                    f'-p "(gpumem > {args.gpumem}) and (gpumodel!=\'k40m\') {host}" ' + \
                    f'-l "{gpus}walltime={args.wall_time}:0:0" -n "{args.run_name}" '


    os.system(oar_specifics +
              f'"./oar_submit_job.sh {resume} ' +
              f'-cr {CONFIG_DIR} ' +
              f'-pf {args.project_config_file} -po \'{args.project_config_overrides}\' ' +
              f'-mf {args.model_config_file} -mo \'{args.model_config_overrides}\'"'
              )

