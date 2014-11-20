#!/usr/bin/env python
"""
   dnolivieri:  14 oct 2014

   The MapReduce version of the RF code for
   MHC.

"""

import collections
import numpy as np
import matplotlib.pyplot as plt
import time
import os, fnmatch
import sys
import itertools
from operator import itemgetter, attrgetter
import math
from Bio import SeqIO
from Bio import Seq
from Bio.SeqRecord import SeqRecord

from scipy import *
import struct
import re
from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence
import json
import cPickle as pickle
from collections import defaultdict
from banyan import *
import multiprocessing
from copy import deepcopy

import timeit


class SimpleMapReduce(object):
    def __init__(self, map_func, partition_func, reduce_func, num_workers=None):
        self.map_func = map_func
        self.partition_func = partition_func
        self.reduce_func = reduce_func

        #self.pool = multiprocessing.Pool(num_workers)
        self.pool = multiprocessing.Pool(4)

    def partition(self, mapped_values):
        partitioned_data= self.partition_func( list(mapped_values))
        return partitioned_data



    def __call__(self, inputs, chunksize=1):
        map_responses = self.pool.map(self.map_func,
                                      inputs,
                                      chunksize=chunksize)

        partitioned_data = self.partition(
            itertools.chain(*map_responses)
            )

        reduced_values = self.pool.map(self.reduce_func,
                                       partitioned_data)
        return reduced_values


# --------------HERE  IS THE Implementation ------

def divide_work(seq):
    def NNchunks(l, n, delta):
        for i in xrange(0, len(l), n):
            if i==0:
                yield (i, i+n,  l[i:i+n])
            else:
                yield (i-1000, i+n,  l[i-1000:i+n])

    return NNchunks(seq, 20000, 1000)



def process_blockNN(d):
    print multiprocessing.current_process().name , d[0], d[1]

    sbar=d[2]
    ix=d[0]

    x=[i.start()+ix for i in re.finditer("AG", str(sbar))]
    y=[i.start()+ix for i in re.finditer("GT", str(sbar))]
    s=[(i,j) for i,j in itertools.product(x,y) if j>i and ( np.abs(i-j)>265  and np.abs(i-j)<285) and ((np.abs(i-j)-2)%3==0) ]
    cnt=0
    cand_list=[]
    cand_seqs=[]
    for i in range(len(s)):
        idx1=s[i][0]-ix   # corrects block start
        idx2=s[i][1]-ix
        test = sbar[idx1+4:idx1+(idx2-idx1)]
        p = test.translate(to_stop=True)
        if (np.abs(int(len(test)/3.)-len(p))<1) and ( (len(p)>88) and (len(p) < 94)):
            cand_list.append(s[i])
            cand_seqs.append(p)
            cnt+=1

    output_list = zip(cand_list, cand_seqs)
    return output_list



def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)


def partition_func(s):
    cand_list = [  x[0] for x in s ]
    cand_seqs = [  x[1] for x in s ]

    sbar = [  x[0] for x in cand_list]
    indices=[]
    for dup in sorted(list_duplicates(sbar)):
        imax=max(dup[1])
        indices.append( [ x for x in dup[1] if x!=imax ] )


    merged = list(itertools.chain.from_iterable(indices))
    cand_list = [i for j, i in enumerate(cand_list) if j not in merged]
    cand_seqs = [i for j, i in enumerate(cand_seqs) if j not in merged]

    return zip(cand_list, cand_seqs)


#rno = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}
#  Add  J==L, B==N
rno = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19, 'J':10, 'B':2 }


qp = open('mats/normalizedAA_Matrix.pkl', 'rb')
normI = pickle.load(qp)

def getPDT3(seq):
    Dvec={}
    cnt=0
    for q in normI:
        sumDseq=0.0
        for i in range(len(seq)-1):
            pi1 = rno[seq[i]]
            pi2 = rno[seq[i+1]]
            sumDseq+= (q[ pi1 ] - q[ pi2 ])*(q[ pi1 ] - q[ pi2 ])
        sumDseq = sumDseq/np.float(len(seq)-1)
        Dvec.update( {str(cnt): sumDseq} )
        cnt+=1
    return Dvec



# ......
def reduce_candidates(s):
    print multiprocessing.current_process().name
    Tx=[]
    seq=str(s[1])
    descObject=PyPro.GetProDes(seq)
    if ('X' not in  seq) and ('Z' not in seq) and ('B' not in seq):
        T = getPDT3(seq)
        Tx = [ T[x]  for x in T.iterkeys() ]

    Y = np.array(Tx)
    return (s[0], s[1], Y)





class MHC1model:
    def __init__(self):
        self.exon_classes= [ 'exon1', 'exon2', 'exon3']
        self.rfmodels = self.get_models()
        self.rec_id=None
        self.rec_name=None
        self.rec_description=None
        self.outfile = None

    def set_record(self, rec_id, rec_name, rec_description):
        self.rec_name=rec_name
        self.rec_id=rec_id
        self.rec_description=rec_description

    def set_outfile(self, outfile):
        self.outfile=outfile


    def get_models(self):
        rfmodels = []
        for exon in self.exon_classes:
            matfile = "mats/trainMat_" + exon + ".pkl"
            fp = open(matfile, 'rb')
            rfmodels.append( pickle.load(fp) )
        return rfmodels


    def exon_probabilities(self, reduced_seqbuffer):
        qbar =  self.sequence_homology(reduced_seqbuffer)
        sbar=[]
        for q in qbar:
            s=(q[0],q[2],q[3][1])
            sbar.append(s)

        ExonList = self.descriminate_by_exon(sbar)
        return ExonList


    def sequence_homology(self, reduced_seqbuffer):
        pos_seqindx = []
        probX=[]
        seqs=[]
        qseq = ()
        seqpos =  [x[0] for x in reduced_seqbuffer if x[2].size!=0 ]
        seqbuffer =  [ str(x[1]) for x in  reduced_seqbuffer if x[2].size!=0   ]
        Y  =  np.array([ x[2] for x in reduced_seqbuffer if x[2].size!=0]   )

        exonType=''
        #print "reduced_seqbuffer=", reduced_seqbuffer
        #print "Y=", Y.shape

        if Y.size!=0:
            for rfM in self.rfmodels:
                probX.append(rfM.predict_proba(Y))

            for j in range(probX[0].shape[0]):
                xbar = np.array([ probX[k][j][1] for k in range(len(probX)) ])
                kmax = np.argmax(xbar)
                ymax = np.max(xbar)

                x=100
                q=seqbuffer[j]
                if 'C' in q[17:23] and 'L' in q[13:20] and 'F' in q[19:25] and 'P' in q[22:28] and 'C' in q[70:x]  and 'W' in q[80:x]:
                    exonType='exon3'
                elif  'H' in q[0:3] and 'G' in q[19:27] and  'D' in q[25:30] and not 'C' in q[5:20]:
                    exonType='exon1'
                elif 'H' in q[1:4] and  'GC' in q[5:27] and  'D' in q[25:30] and 'C' in q[70:82]:
                    exonType='exon2'
                else:
                    exonType=''


                for k in range(len(probX)):
                    print probX[k][j],

                if ymax!=0.0:
                    print ".......", kmax, ymax,   "***", probX[kmax][j], "---", probX[kmax][j][1] > probX[kmax][j][0],  exonType
                else:
                    print

                if (probX[kmax][j][1] > probX[kmax][j][0]):
                    seqs.append( ( seqpos[j],seqbuffer[j], kmax, probX[kmax][j]) )


        return seqs


    def descriminate_by_exon(self, Q):
        s = [x[0] for x in Q]
        eprm = [x[1] for x in Q]
        score= [x[2] for x in Q]

        tree =  SortedSet( s, key=lambda (start, end): (end, (end - start)),updator=OverlappingIntervalsUpdator)

        r = set([])
        tbar=tree.copy()
        for sprm in s:
            a=tree.overlap(sprm)
            b=[s.index(x) for x in tree.overlap(sprm) ]
            c=[eprm[x] for x in b ]
            d=[score[x] for x in b ]
            gX=sorted(list_duplicates(c))

            if len(gX)>0:
                g=sorted(list_duplicates(c))[0][1]
                z=[d[i] for i in g ]
                for i in g:
                    if i!= g[np.argmax(z)]:
                        tree.remove( s[b[i]] )
                        r.update([b[i]])
                else:
                    z=[]

        r_list = [i for j, i in enumerate(Q) if j not in list(r)]
        c=[ x[1] for x in r_list ]

        # IN CLASS Overlaps

        tmpbuff=[]
        rindx_buffer=[]
        for i in range(1,len(c)):
            if (i-1) not in tmpbuff:
                tmpbuff.append(i-1)
            if c[i]== c[i-1]:
                if i not in tmpbuff:
                    tmpbuff.append(i)
            else:
                if len(tmpbuff)>0:
                    z = [ r_list[x][2] for x in tmpbuff ]
                    imax=np.argmax(z)
                    jmax=tmpbuff[imax]

                    for j in tmpbuff:
                        if j!=jmax:
                            rindx_buffer.append(j)

                tmpbuff=[]


            lastI=i



        merged = rindx_buffer
        Q = [i for j, i in enumerate(r_list) if j not in merged]

        ## Out-Of-Class Overlaps

        s = [ x[0] for x in Q ]
        eprm = [x[1] for x in Q]
        score= [x[2] for x in Q]

        tree =  SortedSet( s, key=lambda (start, end): (end, (end - start)),updator=OverlappingIntervalsUpdator)
        r = set([])
        for sprm in s:
            a=tree.overlap(sprm)
            b=[s.index(x) for x in tree.overlap(sprm) ]
            c=[eprm[x] for x in b ]
            d=[score[x] for x in b ]

            if len(b)>1:
                for i in range(len(d)):
                    if i!= np.argmax(d):
                        tree.remove( s[b[i]] )
                        r.update([b[i]])
                    else:
                        z=[]

        cand_list = [i for j, i in enumerate(Q) if j not in list(r)]


        ## -----------------------
        Z=['A','B','C']
        pattern = [ Z[x[1]] for x in cand_list ]
        exonString=''.join(map(str, pattern))
        print exonString
        x=[i.start() for i in re.finditer("ABC", exonString)]

        print x
        ExonList=[]
        for ix in x:
            print "-----exon1 - exon2 - exon3---------"
            print ix, cand_list[ix:ix+3]

            ## test the mutual distances.
            xp = [ q[0][0] for q in cand_list[ix:ix+3] ]
            print xp,  "e2-e1=",xp[1]-xp[0], "e3-e2=", xp[2]-xp[1]

            if ((xp[1]-xp[0])<5000) and ((xp[2]-xp[1])<5000):
                ExonList.append((cand_list[ix: ix+3]) )

        return ExonList


    def MHC1_exon_model( self, mcnt, seq, strand, AllExons):
        print "len(AllExons)=", len(AllExons)
        MHCList=[]
        for p in AllExons:
            indx1= p[0][0]
            indx2= p[1][0]
            indx3= p[2][0]
            exon1 = seq[indx1[0]+2: indx1[0]+ (indx1[1] - indx1[0]) ]
            exon2 = seq[indx2[0]+2: indx2[0]+ (indx2[1] - indx2[0]) ]
            exon3 = seq[indx3[0]+2: indx3[0]+ (indx3[1] - indx3[0]) ]
            print p

            print   "----exon1--------------", "len(exon1)=", len(exon1)
            print  seq[indx1[0]: indx1[0]+ 10], ".....", seq[ indx1[0]+(indx1[1] - indx1[0])-10:  indx1[0]+(indx1[1] - indx1[0])+2 ]
            print  "  ",seq[indx1[0]+2: indx1[0]+ 10], ".....", seq[ indx1[0]+(indx1[1] - indx1[0])-10:  indx1[0]+(indx1[1] - indx1[0]) ]

            print   "----exon2--------------", "len(exon2)=", len(exon2)
            print  seq[indx2[0]: indx2[0]+ 10], ".....", seq[ indx2[0]+(indx2[1] - indx2[0])-10:  indx2[0]+(indx2[1] - indx2[0])+2 ]
            print  "  ",seq[indx2[0]+2: indx2[0]+ 10], ".....", seq[ indx2[0]+(indx2[1] - indx2[0])-10:  indx2[0]+(indx2[1] - indx2[0]) ]

            print   "----exon3--------------", "len(exon3)=", len(exon3)
            print  seq[indx3[0]: indx3[0]+ 10], ".....", seq[ indx3[0]+(indx3[1] - indx3[0])-10:  indx3[0]+(indx3[1] - indx3[0])+2 ]
            print  "  ",seq[indx3[0]+2: indx3[0]+ 10], ".....", seq[ indx3[0]+(indx3[1] - indx3[0])-10:  indx3[0]+(indx3[1] - indx3[0]) ]

            RNA = exon1+exon2+exon3
            MHC = RNA[2:-1].translate()
            #MHC = RNA[:].translate()
            print "MHC=", MHC
            print
            exon1_ival = (indx1[0], indx1[1])
            exon2_ival = (indx2[0], indx2[1])
            exon3_ival = (indx3[0], indx3[1])
            exons_ivals=( exon1_ival, exon2_ival, exon3_ival)
            MHCList.append( (MHC, self.rec_id, self.rec_description, exons_ivals ) )


        print "len(MHCList)=", len(MHCList)
        for p in MHCList:
            print "OUTPUT MHC=", str(p[0])
            rec_id = p[1].split("|")[3]
            t= p[2].split("|")[4].strip().split(" ")
            rec_name=t[0]+"_"+t[1]
            recordB=SeqRecord(p[0], id = "MHCI-"+str(mcnt)+"-RF-"+rec_id+"|"+rec_name+"|"+str(strand), description=str(p[3]) )
            self.outfile.write(recordB.format("fasta"))
            mcnt+=1

        return mcnt




class RunMHCpredict:
    def __init__(self, S, desc_method , mammalList, mapper):
        strand=1
        self.S = S
        self.desc_method= desc_method
        self.mammalList = mammalList
        self.mapper = mapper

        qp = open('mats/normalizedAA_Matrix.pkl', 'rb')
        self.normI = pickle.load(qp)
        self.predicted_seqs = []
        self.contigs = self.get_contigs(mammalList[0])
        self.analyze_files(mammalList)

    def get_contigs(self, mammal):
        contigs=[]
        fp = open(self.S[mammal]["contigs"], "r")
        for lk in fp:
            contigs.append(lk.strip())
        fp.close()
        #print contigs
        return contigs


    def analyze_files(self, mammalList):
        #for mammal in self.T.iterkeys():
        Mmodel = MHC1model()
        for mammal in mammalList:
            fbar= self.S[mammal]["WGS"]
            print fbar
            outFile = fbar.replace(".fasta", "_outRF.fasta")
            ofile = open(outFile,"w")
            Mmodel.set_outfile( ofile )
            start_time = timeit.default_timer()
            gene_cnt=0
            for strand in [1, -1]:
                print "STRAND=", strand
                for record in SeqIO.parse(fbar, "fasta"):
                    if ( record.id.split("|")[3] not in self.contigs):
                        continue

                    print record.id.split("|")[3]
                    if strand == 1:
                        seq=record.seq
                    else:
                        seq=record.seq.reverse_complement()

                    print record.id, record.name
                    Mmodel.set_record(record.id, record.name, record.description)
                    seq_size=len(seq)

                    res= self.mapper( divide_work(seq) )
                    #print "res=", res

                    #Mmodel.sequence_homology(res)
                    Elist=Mmodel.exon_probabilities(res)
                    gene_cnt=Mmodel.MHC1_exon_model(gene_cnt, seq, strand, Elist)
                    res=None
                    Elist=None

            ofile.close()
            elapsed = timeit.default_timer() - start_time
            print "ELAPSED TIME =", elapsed



if __name__ == '__main__':
    import operator


    mapper = SimpleMapReduce(process_blockNN, partition_func, reduce_candidates)

    MHC1_prediction = 'MHC1_Prediction.json'
    json_data=open( MHC1_prediction )
    S = json.load(json_data)
    json_data.close()


    mlist = []
    mlist.append(sys.argv[1])
    M =  RunMHCpredict(S,  desc_method='PDT', mammalList= mlist, mapper=mapper )


