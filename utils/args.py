import argparse

def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= federated learning parameters ========================
    parser.add_argument('--seed', type=int, default=0,
                        help="exp mark")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--ul_mode', choices=['none',
                                              'ul_samples', 'ul_samples_backdoor', 'retrain_samples',
                                              'ul_class','retrain_class',
                                              'amnesiac_ul_samples','amnesiac_ul_class'
                                              ],
                         default='ul_class', type=str,
                         help='which unlearning scheme we use') 
    parser.add_argument('--num_ul_users', type=int, default=1,
                        help="number of unlearning users")
    parser.add_argument('--external_ul_dataloader', type=str, default=None,
                        help='saved dataloader used only when --num_ul_users is 0')
    parser.add_argument('--ul_class_id', type=int, default=9,
                        help="id of unlearned class")
    parser.add_argument('--samples_per_user', type=int, default=5000,
                        help="number of users: K")
    parser.add_argument('--persample_bs', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--defense', type=str, default="none",
                        help="defense scheme")
    parser.add_argument('--d_scale', type=float, default=0.0,
                        help="number of users: K")
    parser.add_argument('--save_dir', type=str, default='./outputs',
                        help='saving path')
    parser.add_argument('--log_folder_name', type=str, default='./outputs',
                        help='saving path')
    parser.add_argument('--proportion', type=float, default=0.00,
                        help="the proportion of UL samples")
    parser.add_argument('--frac', type=float, default=1,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr_outer', type=float, default=1,
                        help="learning rate")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate for inner update")
    parser.add_argument('--lr_up', type=str, default='common',
                        help='optimizer: [common, milestone, cosine]')
    parser.add_argument('--schedule_milestone', type=list, default=[225,325],
                         help="schedule lr")
    parser.add_argument('--gamma', type=float, default=0.99,
                         help="exponential weight decay")
    parser.add_argument('--iid',type=int, choices=[0, 1], default=1,
                        help='dataset is split iid or not')
    parser.add_argument('--fine_tune_mode',type=int, default=0,
                        help='the model is fine-tuned or not')
    parser.add_argument('--beta', type=float, default=1,
                        help='Non-iid Dirichlet param')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='communication round')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of client sampling we use') 
    parser.add_argument('--data_augment', action='store_true', default=True,
                        help='data_augment')
    parser.add_argument('--lira_attack', action='store_true', default=True,
                        help='lira_attack')
    parser.add_argument('--cosine_attack', action='store_true', default=True,
                        help='cosine_attack')
    parser.add_argument('--class_prune_sparsity', type=float, default=0.05,
                        help='class_prune_sparsity')
    parser.add_argument('--class_prune_target', type=int, default=9,
                        help='class_prune_target')
    parser.add_argument('--ul_client_gamma', type=float, default=0.5,
                        help='ul_client_gamma')
    parser.add_argument('--ul_samples_alpha', type=float, default=0.9,
                        help='ul_samples_alpha')
    parser.add_argument('--recovery_epochs', type=int, default=1,
                        help='FedUL independent retained-set recovery epochs after unlearning')
    parser.add_argument('--recovery_lr', type=float, default=None,
                        help='FedUL independent retained-set recovery learning rate; default uses --lr')
    parser.add_argument('--recovery_freeze_aux', type=int, default=1,
                        help='freeze FedUL independent auxiliary branch during recovery')
    parser.add_argument('--recovery_objective', type=str, default='unlearn',
                        choices=['unlearn', 'learning'],
                        help='FedUL recovery objective: final unlearn output or learning branch logits')
    parser.add_argument('--recovery_accept_ul_drop', type=float, default=0.02,
                        help='max accepted UL-effect drop when keeping FedUL recovery')
    parser.add_argument('--disable_cne', action='store_true', default=False,
                        help='disable calibrated negative expert for FedUL ablation')
    parser.add_argument('--disable_gra', action='store_true', default=False,
                        help='disable gated retain routing for FedUL ablation')
    parser.add_argument('--amnesiac_start_epoch', type=int, default=None,
                        help='override the first epoch used when subtracting Amnesiac sensitive updates')
    
    
    # ============================ Model arguments ===================================
    parser.add_argument('--model_name', type=str, default='alexnet', choices=['lenet','alexnet','resnet18','resnet34'],
                        help='model architecture name')
    # ========== 【添加新参数】==========
    parser.add_argument('--model_variant', type=str, default='original',
                        choices=['original', 'independent'],
                        help='模型变体: original (原始FedAU) 或 independent (独立双分支)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'celeba'],
                        help="name of dataset")
    parser.add_argument('--celeba_attr', type=str, default='Smiling',
                        help='CelebA attribute for binary mode, or comma-separated attributes for attrs10 mode')
    parser.add_argument('--celeba_target_mode', type=str, default='attrs10',
                        choices=['binary', 'attrs10'],
                        help='CelebA target mode: binary attribute or 10-class attribute combinations')
    
    parser.add_argument('--data_root', default='./data',
                        help='dataset directory')
    parser.add_argument('--pretrain_model_root', default='FedAU/log_test_pretrain',
                        help='the saved pre-trained model directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'auto'],
                        help='execution device; cuda preserves the reference experiment behavior')
    parser.add_argument('--gpu', default='0', type=str,
                        help='legacy GPU label retained for command compatibility')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--bp_interval', default=30, type=int, help='interval for starting bp the local part')
    parser.add_argument('--log_interval', default=1, type=int,
                        help='interval for evaluating loss and accuracy')

    parser.add_argument("--sigma_sgd",
        type=float,
        default=0.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
    "--grad_norm",
    type=float,
    default=1e4,
    help="Clip per-sample gradients to this norm",
    )

    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')

    # =========================== DP ===================================
    parser.add_argument('--dp', action='store_true', default=False,
                        help='whether dp')

    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')



    # =========================== Robustness ===================================
    parser.add_argument('--pruning', action='store_true')
    parser.add_argument('--percent', default=5, type=float)

    # parser.add_argument('--im_balance', action='store_true', default=False,
    #                     help='whether im_balance')
    
    args = parser.parse_args()

    return args
