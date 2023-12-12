import argparse
import src.data as data
import src.trainer as trainer
import src.timer as timer
import src.saver as save

def main(args)->None:
    my_timer = timer.Timer()
    training_data = data.Data()
    config_data = data.ConfigData("config.yaml")
    training_data.configure(config_data)
    training_data.use_random_points()

    if(args.use_static_data):
        training_data.use_static_points()

    my_timer.start()

    hilbert_space, optimal_c, risk, emperical_risk = trainer.optimize(training_data, config_data)

    print(my_timer)

    if(args.save):
        save.plot_support_vector_machine(training_data, optimal_c)
        #save.plot_error(hilbert_space, risk, emperical_risk)
        #save.plot_everything(training_data, optimal_c, hilbert_space, risk, emperical_risk)

    return


if("__main__" == __name__):
    parser = argparse.ArgumentParser(description="generate support vector machine")
    parser.add_argument("--use_static_data", action="store_true", help="set True for using static data defined in config.yaml")
    parser.add_argument("--save", action="store_true", help="set True for building visualizer for separating hyperplane")
    args = parser.parse_args()
    main(args)
