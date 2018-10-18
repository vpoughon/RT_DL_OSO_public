import numpy as np
import horovod.keras as hvd
import sys
import os

from .cnes_generator_10m import CnesGeneratorSentinel
from mpi4py import MPI

class CnesGen10mUtilHvd:
    cnes_gen = None
    patch_size = 0

    def __init__(self, gen, patch_size):
        self.cnes_gen = gen
        self.patch_size = patch_size

    def generate_train_patch_using_sharing(self, batch_size):
        comm = MPI.COMM_WORLD
        if hvd.rank() == 0:
            batch_patch_info = self.cnes_gen.choose_patches_for_iteration(batch_size * hvd.size())
            # batch_img, batch_gt = cnes_gen.generate_train_patch_fast(BATCH_SIZE, batch_patch_info)
            transfers = self.get_batch_sharing_solution(batch_patch_info)

            for k in range(1, hvd.size()):
                comm.send(batch_patch_info, dest=k, tag=1001)
                comm.send(transfers, dest=k, tag=1002)
        else:
            batch_patch_info = comm.recv(source=0, tag=1001)
            transfers = comm.recv(source=0, tag=1002)

#            batch_patch_info = np.zeros((6, batch_size * hvd.size()), np.int32)
#            transfers = np.zeros((hvd.size(), hvd.size()), np.int32)

#        batch_patch_info = hvd.broadcast(batch_patch_info, root_rank=0, name="BATCH_PATCH_INFO")
#        transfers = hvd.broadcast(transfers, root_rank=0, name="TRANSFERS")

#        batch_patch_info = comm.bcast(batch_patch_info, root=0)
#        transfers = comm.bcast(transfers, root=0)

        return self.get_batch_using_sharing(batch_size, batch_patch_info, transfers)

    def get_batch_using_sharing(self, batch_size, batch_patch_info, transfers):

        sys.stdout.flush()
        sys.stderr.flush()

        comm = MPI.COMM_WORLD
        comm.barrier()

        dbginfo = "CNES_DBG_BATCH_SHARING" in os.environ

        local_batch_patch_info = batch_patch_info[:,batch_patch_info[0,:] == hvd.rank()]

        if dbginfo:
            print("Rank {} has {} images".format(hvd.rank(), local_batch_patch_info.shape[1]))
        all_img_batch, all_gt_batch = self.cnes_gen.generate_train_patch_fast(batch_size, local_batch_patch_info)
        batch_idx = batch_size
        n_sent = 0

        patch_memory_size = self.cnes_gen.get_num_bands() * self.patch_size * self.patch_size * 4
        gt_patch_memory_size = len(self.cnes_gen.get_class_ids()) * self.patch_size * self.patch_size * 4
        max_message_size = (patch_memory_size + gt_patch_memory_size) * batch_size * hvd.size()
        
        send_requests = []
        recv_requests = []
        #need to fill missing data in batch or send surplus
        for dest_node in range(transfers.shape[0]):
            order = transfers[dest_node, hvd.rank()]
            if order < 0:
                n_to_send = -order
                data = all_img_batch[batch_idx:(batch_idx+n_to_send), :, :, :]
                gt_data = all_gt_batch[batch_idx:(batch_idx+n_to_send), :, :, :]
                if dbginfo:
                    print("Rank {} sending {} images (indexes {}: {}),  tensor shape {} to rank {}".format(hvd.rank(),
                                                                                                           n_to_send,
                                                                                                           batch_idx,
                                                                                                           (batch_idx + n_to_send),
                                                                                                           data.shape, dest_node))
                    sys.stdout.flush()
                #req = comm.isend([data, gt_data], dest=dest_node, tag=1000)
                #send_requests.append(req)
                batch_idx += n_to_send
                n_sent += n_to_send
                # ajout VP
                comm.Send(data, dest=dest_node, tag=1000)
                comm.Send(gt_data, dest=dest_node, tag=2000)
                # fin ajout VP
            elif order > 0:
                if dbginfo:
                    print("Rank {} receives {} images from rank {}".format(hvd.rank(), order, dest_node))
                    sys.stdout.flush()
                    #print((order, all_img_batch.shape[1], all_img_batch.shape[2], all_img_batch.shape[3]))
                #req = comm.irecv(max_message_size, source=dest_node, tag=1000)
                #recv_requests.append(req)
                # ajout VP, to adjust accurately the buffer size to the exchanged data size. max_message_size can lead to Overflow error
                data = np.empty((order, all_img_batch.shape[1], all_img_batch.shape[2], all_img_batch.shape[3]), dtype=all_img_batch.dtype)
                gt_data = np.empty((order, all_gt_batch.shape[1], all_gt_batch.shape[2], all_gt_batch.shape[3]), dtype=all_gt_batch.dtype)
                comm.Recv(data, source=dest_node, tag=1000)
                comm.Recv(gt_data, source=dest_node, tag=2000)
                all_img_batch = np.concatenate((all_img_batch, data), axis=0)
                all_gt_batch = np.concatenate((all_gt_batch, gt_data), axis=0)
                # fin ajout VP

        #for req in send_requests:
            #req.wait()

        #for req in recv_requests:
            #data_all = req.wait()
            #if data_all is not None:
                #all_img_batch = np.concatenate((all_img_batch, data_all[0]), axis=0)
                #all_gt_batch = np.concatenate((all_gt_batch, data_all[1]), axis=0)
                #if dbginfo:
                    #print("Rank {} : adding images size {} to batch ".format(hvd.rank(), data_all[0].shape))
            #else:
                #raise Exception("Rank {} : supposed to get data but nothing received".format(hvd.rank()))
        
        if n_sent > 0:
            all_img_batch = all_img_batch[0:(all_img_batch.shape[0]-n_sent)]
            all_gt_batch = all_gt_batch[0:(all_gt_batch.shape[0] - n_sent)]

        if all_img_batch.shape[0] != batch_size:
            raise Exception("Local batch does not have the right size after redistribution {} vs {}".format(all_img_batch.shape[0], batch_size))
        if all_gt_batch.shape[0] != batch_size:
            raise Exception("Local GT batch does not have the right size after redistribution {} vs {}".format(all_gt_batch.shape[0], batch_size))

        return all_img_batch, all_gt_batch

    def get_batch_sharing_solution(self, batch_patch_info):
        n_workers = hvd.size()
        n_imgs_per_worker = batch_patch_info.shape[1] // hvd.size()

        worker_batch_delta = np.zeros((hvd.size(), 2), np.int32)
        for wi in range(n_workers):
            worker_batch_delta[wi, 0] = np.sum(batch_patch_info[0, :] == wi) - n_imgs_per_worker
            worker_batch_delta[wi, 1] = wi

        sw = worker_batch_delta[worker_batch_delta[:, 0].argsort(), :] # sorted by decreasing nb of missing patchs
 #       print("***SHARING SOLUTION***")
 #       print("INITIAL OFFERINGS")
 #       print(sw)
        transfers = np.zeros((n_workers, n_workers), np.int32)
        i = 0
        j = n_workers - 1
        while i < j and sw[i, 0] < 0: # where there are missing patches for a worker
            if sw[i, 0] < 0 and sw[j, 0] > 0: # if patches missing for a worker and too much for another
                init_i = sw[i,0]
                init_j = sw[j,0]
                if -sw[i,0] < sw[j,0]:  #worker j having images can fullfill request of i and still have some images left
                    transfers[sw[i,1], sw[j,1]] = sw[i,0]
                    transfers[sw[j,1], sw[i,1]] = -sw[i,0]
                    sw[j,0] += sw[i,0]
                    sw[i,0] = 0
                    i += 1
                else:   #worker i having images can can get all images of i and (may) still need some more
                    transfers[sw[i,1], sw[j,1]] = -sw[j,0]
                    transfers[sw[j,1], sw[i,1]] = sw[j,0]
                    sw[i,0] += sw[j,0]
                    sw[j,0] = 0
                    if -init_i == init_j:  #if both are fullfilled continue
                        i += 1
                    j -= 1

        if not np.sum(sw[:,0]) == 0:
            raise Exception("Error in sharing solution, check source code !!!!")
        return transfers
#        print("RESOLVED OFFERINGS")
#        print(sw)
#        print("TRANSFERS")
#        print(transfers)

    def print_global_running_stats(self):
        stats = self.cnes_gen.get_running_stats()

        CLASS_ID_SET = self.cnes_gen.get_class_ids()

        print("stats at rank {} : {}".format(hvd.rank(), stats))

        stats_mat = np.zeros((len(CLASS_ID_SET) + 1, 2), np.float32)
        stats_mat[0, 1] = stats[0]
        idx = 1
        for cid in CLASS_ID_SET:
            stats_mat[idx, 0] = cid
            if cid in stats:
                stats_mat[idx, 1] = stats[cid]
            idx += 1

        print("Gathering stats from all MPI instances, rank {}".format(hvd.rank()))
        all_stats = hvd.allgather(stats_mat)  # comm.gather(stats, root=0)
        total_px = 0

        if hvd.rank() == 0:
#            print("Epoch {} class freqs:".format(self.epoch))
            class_stats = {class_id: 0 for class_id in CLASS_ID_SET}
            for class_id in CLASS_ID_SET:
                # print("Data for class {}: {}".format(class_id, all_stats[all_stats[:,0] == class_id, :]))
                px_class = np.sum(all_stats[all_stats[:, 0] == class_id, 1])
                class_stats[class_id] += px_class
                total_px += px_class

            non_annot_px = np.sum(all_stats[all_stats[:, 0] == 0, 1])
            total_px += non_annot_px
            print("Non annotated pixels : {}".format(non_annot_px))
            for class_id in class_stats:
                print("Class {} count = {}, freq {:.5f}%".format(class_id, class_stats[class_id],
                                                                 class_stats[class_id] / total_px * 100))
