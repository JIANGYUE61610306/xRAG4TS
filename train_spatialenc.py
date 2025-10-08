from util import *
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from spatialencoder import tsMAE_trainer, TSMAE
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='./data/SG')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--history_seq_len", type=int, default=288)
parser.add_argument('--input_seq_len', default=12, type=int)
parser.add_argument('--output_seq_len', default=12, type=int)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio during pre-training')
parser.add_argument('--patch_size', type=int, default=12)
parser.add_argument('--embed_dim', type=int, default=96, help='Embedding/hidden dim for transformer')
parser.add_argument('--encoder_depth', type=int, default=4, help='Number of transformer blocks in encoder')
parser.add_argument('--decoder_depth', type=int, default=1, help='Number of transformer blocks in decoder')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in TSMAE')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_token', type=int, default=168)
parser.add_argument('--in_channel', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=4, help='The width of mlp w.r.t the transformer')
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--pretrain_epoch', type=int, default=50, help='Pre-train epoch for TSMAE')
parser.add_argument("--print_every", type=int, default=200)
parser.add_argument('--expid', type=str, default='test')
parser.add_argument('--model', type=str, default='TSMAE', help='[TSMAE, TSFormer]')
parser.add_argument("--pretrained_model", type=str,help='Load a pre-trained model')
parser.add_argument("--data_number", type=int, default=0, help='The number of data used to pretrain. in days')
args = parser.parse_args()

parser.add_argument("--query_data_path", type=str, default='./data/NOTTINGHAM',
                    help='Path to NOTTINGHAM dataset folder for query embeddings')

def main(args):
    args.num_token = int(args.history_seq_len / 12)
    train_dataset = TimeSeriesForecastingDataset(args.data_path+'/data_in288_out12.pkl', 
        args.data_path+'/index_in288_out12.pkl', mode = 'train', data_number = args.data_number)
    val_dataset = TimeSeriesForecastingDataset(args.data_path+'/data_in288_out12.pkl', 
        args.data_path+'/index_in288_out12.pkl', mode = 'valid')
    test_dataset = TimeSeriesForecastingDataset(args.data_path+'/data_in288_out12.pkl', 
        args.data_path+'/index_in288_out12.pkl', mode = 'test')
    scaler_pkl = load_pkl(args.data_path+'/scaler_in288_out12.pkl')
    mean, std = scaler_pkl['args']['mean'], scaler_pkl['args']['std']
    print(mean,std)
    scaler = Scaler(mean, std)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size)
    save_loader = DataLoader(train_dataset, batch_size = args.batch_size)


    #load query data
    query_dataset = TimeSeriesForecastingDataset(
        os.path.join(args.query_data_path, 'data_in288_out12.pkl'),
        os.path.join(args.query_data_path, 'index_in288_out12.pkl'),
        mode='train',
        data_number=3
    )
    query_loader = DataLoader(query_dataset, batch_size = args.batch_size)



    engine = tsMAE_trainer(args, scaler)
    print("Pretraining TSMAE on %s data" % args.data_path)
    print("Train data length %d" % len(train_dataset))
    print("Val data length %d" % len(val_dataset))
    print("Test data length %d" % len(test_dataset))
    if 'SG' in args.data_path:
        save_tsmae = 'garage_tsmae/SG/' + args.expid + '/'
        if not os.path.exists(save_tsmae):
            os.makedirs(save_tsmae)
    if args.pretrained_model is not None:
        print("Load a pretrained model from %s" % args.pretrained_model)
        sd = torch.load(args.pretrained_model, map_location = args.device)
        try:
            engine.tsmae_model.load_state_dict(sd['model_state_dict'])
        except:
            engine.tsmae_model.load_state_dict(sd)
        val_maes = []
        val_rmses = []
        val_mapes = []
        t1 = time.time()
        for y, x in val_loader:
            metrics = engine.eval_pretrain(x)
            val_maes.append(metrics[0])
            val_rmses.append(metrics[1])
            val_mapes.append(metrics[2])
        t2 = time.time()
        print("Evaluate the loaded model on validation data")
        print("Time %.4fs, mae %.4f, rmse %.4f, mape %.4f"%(t2-t1, np.mean(val_maes), np.mean(val_rmses), np.mean(val_mapes)))
        test_maes = []
        test_rmses = []
        test_mapes = []
        t1 = time.time()
        for y, x in test_loader:
            metrics = engine.eval_pretrain(x)
            test_maes.append(metrics[0])
            test_rmses.append(metrics[1])
            test_mapes.append(metrics[2])
        t2 = time.time()
        print("Evaluate the loaded model on test data")
        print("Time %.4fs, mae %.4f, rmse %.4f, mape %.4f"%(t2-t1, np.mean(test_maes), np.mean(test_rmses), np.mean(test_mapes)))

    
    val_loss = []
    train_loss = []
    train_time = []
    val_time = []
    for ep in range(1, args.pretrain_epoch+1):
        epoch_trainloss = []
        epoch_trainrmse = []
        epoch_trainmape = []
        epoch_valloss = []
        epoch_valrmse = []
        epoch_valmape = []
        s1 = time.time()
        for i, (y, x) in enumerate(train_loader):
            # print(i, time.time() - s1)
            # print('x', x.shape)
            # print('y', y.shape)
            # reconstructed, label = model(x[:,:,:,:2])
            # print(x[0,:288,0,1] * 288)
            # print(x[0,:288,0,2])
            # break
            if args.history_seq_len != 2016:
                x = x[:,-args.history_seq_len:,:,:]
            # print('x shape is: *******', x.shape)
            metrics = engine.pre_train(x)
            # print('reconstructed', reconstructed.shape)
            # print('label', label.shape)
            epoch_trainloss.append(metrics[0])
            epoch_trainrmse.append(metrics[1])
            epoch_trainmape.append(metrics[2])
            if i % args.print_every == 0:
                print("Pretrain Epoch %d, Iter %d, MAE %.4f, RMSE %.4f, MAPE %.4f, Time spent %.4fs" % (ep, i, metrics[0], metrics[1], metrics[2], time.time() - s1))
        engine.scheduler.step()
        s2 = time.time()
        train_time.append(s2-s1)
        train_loss.append(np.mean(epoch_trainloss))
        print("Pretrain Epoch %d, train MAE %.4f, train RMSE %.4f, train MAPE %.4f, Time %.4fs, current lr %.5f" % (ep, train_loss[-1], np.mean(epoch_trainrmse), np.mean(epoch_trainmape), s2-s1, engine.scheduler.get_last_lr()[0]))
        t1 = time.time()
        for y, x in val_loader:
            if args.history_seq_len != 2016:
                x = x[:,-args.history_seq_len:,:,:]
            # print('x shape is: *******', x.shape)
            metrics = engine.eval_pretrain(x)
            epoch_valloss.append(metrics[0])
            epoch_valrmse.append(metrics[1])
            epoch_valmape.append(metrics[2])
        t2 = time.time()
        val_time.append(t2-t1)
        val_loss.append(np.mean(epoch_valloss))
        print("Pretrain Epoch %d, val MAE %.4f, val RMSE %.4f, val MAPE %.4f, Time %.4fs" % (ep, val_loss[-1], np.mean(epoch_valrmse), np.mean(epoch_valmape), t2-t1))
        torch.save(engine.tsmae_model.state_dict(), save_tsmae + 'epoch_%d_%.4f.pth' % (ep, val_loss[-1]))
    best_val_idx = np.argmin(val_loss)
    print("Pre-training finish. ")
    print("Average epoch train time %.4fs, val time %.4fs" % (np.mean(train_time), np.mean(val_time)))
    print("The epoch with best pre-training validation loss is %d, best loss %.4f" % (best_val_idx, val_loss[best_val_idx]))
    engine.tsmae_model.load_state_dict(torch.load(save_tsmae+'epoch_%d_%.4f.pth' % (best_val_idx+1, val_loss[best_val_idx])))
    save_path = save_tsmae+'best_pretrained_%s_%.4f.pth' % (args.expid, val_loss[best_val_idx])
    torch.save(engine.tsmae_model.state_dict(), save_tsmae + 'best_pretrained_%s_%.4f.pth' % (args.expid, val_loss[best_val_idx]))

    print('Start building doc repo...')
    save_doc = []
    ori_input = []

    #save train embeddings
    for y, x in save_loader:
            if args.history_seq_len != 2016:
                x = x[:,-args.history_seq_len:,:,:]
            # print('x shape is: *******', x.shape)
            doc = engine.save_doc(x)
            save_doc.append(doc)
            ori_input.append(x)
    print('Embedding finished, start saving doc embeddings...')
    print('save_doc length is: ', len(save_doc))
    save_doc_np = [tensor.cpu().numpy() for tensor in save_doc]
    ori_input_np = [tensor.cpu().numpy() for tensor in ori_input]
    # Convert to a NumPy array with object dtype
    save_doc_np_array = np.array(save_doc_np, dtype=object)
    ori_input_np_array = np.array(ori_input_np, dtype=object)
    # Save it — now it works
    np.save('save_doc.npy', save_doc_np_array)
    np.save('ori_input.npy', ori_input_np_array)

    #save query embeddings
    query_doc = []
    query_input = []
    for y, x in query_loader:
            if args.history_seq_len != 2016:
                x = x[:,-args.history_seq_len:,:,:]
            # print('x shape is: *******', x.shape)
            doc = engine.save_doc(x)
            query_doc.append(doc)
            query_input.append(x)
    print('Embedding finished, start saving doc embeddings...')
    print('save_doc length is: ', len(query_doc))
    query_doc_np = [tensor.cpu().numpy() for tensor in query_doc]
    query_input_np = [tensor.cpu().numpy() for tensor in query_input]
    query_doc_np_array = np.array(query_doc_np, dtype=object)
    query_input_np_array = np.array(query_input_np, dtype=object)
    # print(query_input_np_array.shape)
    # print(query_doc_np_array.shape)
    np.save('query_doc.npy', query_doc_np_array)
    np.save('query_input.npy', query_input_np_array)



    print("Saving the best pre-trained TSMAE to %s" % save_path)
    np.save(save_tsmae+'train_loss.npy', arr = np.array(train_loss))
    np.save(save_tsmae+'val_loss.npy', arr = np.array(val_loss))

if __name__ == '__main__':
    main(args)
