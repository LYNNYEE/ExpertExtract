
'''
使用LSTM-CRF的版本 crf的结果影响cost  transfer可以自动生成和改变
'''
import tensorflow as tf
import numpy as np
import re
import os
import codecs
from lxml import etree
from lxml.html import clean  
import math
import nltk
import pickle
import scipy
from scipy.sparse import csr_matrix
from collections import Counter
import copy
import time
#import Tagger
import lstm_tagger
from six.moves import xrange
import math



class LSTM_CRF():
	def __init__(self,batch_size,in_vector_len,out_class_len,do_train):
		self.batch_size = batch_size
		#self.time_step = time_step
		self.in_vector_len = in_vector_len
		self.out_class_len = out_class_len
		self.do_train = do_train

	def build_model(self):
		self.cell_num = 128
		learning_rage = 0.001

		self.X = tf.placeholder(tf.float32,[self.batch_size,None,self.in_vector_len])
		self.Y = tf.placeholder(tf.int32,[self.batch_size,None])
		self.s_length = tf.placeholder(tf.int32,shape=[self.batch_size])

		weights = {
		    # 神经网络输出: [textlen,cellnum]*[cell_num,out_class_len] + [out_class_len] = [textlen,out_class_len]
		    'out': tf.Variable(tf.random_normal([self.cell_num,self.out_class_len]))
		}
		biases = {
		    'out': tf.Variable(tf.constant(0.1, shape=[self.out_class_len,]))
		}

		self.hidden_state = tf.placeholder(tf.float32,[self.batch_size, None])
		self.current_state = tf.placeholder(tf.float32,[self.batch_size, None])
		self.cell_state = self.current_state,self.hidden_state

		def RNN(X, weights, biases):
			cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.cell_num,forget_bias=1.0,state_is_tuple=True)
			self.cell_state = cell.zero_state(self.batch_size, dtype=tf.float32)
			#outputs : [batch_size,time_steps,cell_num]
			outputs, cell_final_state  = tf.nn.dynamic_rnn(cell, X, initial_state=self.cell_state,sequence_length=self.s_length,time_major=False)
			outputs = tf.reshape(outputs,[-1,self.cell_num])
			#outputs = tf.reshape(outputs,[-1,cell_num])
			#result [textlen,out_class_len]
			result = tf.matmul(outputs,weights['out']) + biases['out']
			result = tf.reshape(result,[self.batch_size,-1,self.out_class_len])
			return result,cell_final_state

		#pred 
		self.pred,self.cell_final_state = RNN(self.X,weights,biases)
		self.transition_params = tf.placeholder(tf.float32,[self.out_class_len,self.out_class_len])

		#增加上一分组最后一个字的状态
		self.prestep_laststate = tf.placeholder(tf.int32,shape=[self.batch_size])
		self.fixed_preds = [[] for i in xrange(self.batch_size)]
		for b in xrange(self.batch_size):
			first_state = tf.add(self.pred[b][0],self.transition_params[self.prestep_laststate[b]])
			first_state = tf.expand_dims(first_state,0)
			fixed_pred = tf.concat([first_state,self.pred[b][1:]],0)
			self.fixed_preds[b] = fixed_pred
		self.fixed_preds = tf.convert_to_tensor(self.fixed_preds,dtype=tf.float32)
		
		if self.do_train:
			log_likelihood,self.transition_params = tf.contrib.crf.crf_log_likelihood(self.fixed_preds,self.Y, self.s_length,self.transition_params)
			self.pred,self.best_scores = tf.contrib.crf.crf_decode(self.fixed_preds,self.transition_params,self.s_length)
			self.t_length = tf.placeholder(tf.int32,shape=[self.batch_size])
			self.state_rates = tf.placeholder(tf.float32,[self.out_class_len])
			c_weights = []
			for b in range(self.batch_size):
				c_weights.append(tf.div(tf.cast(self.t_length[b],dtype=tf.float32),tf.cast(tf.reduce_sum(self.t_length),dtype=tf.float32)))
			#根据实际所属的类别 进行加权 避免过少出现的状态在误差修正的时候会考虑不到
			s_weights = []
			for b in range(self.batch_size):
				valid_len = tf.range(0,self.s_length[b])
				valid_Y = tf.gather(self.Y[b],valid_len)
				count_y,_,count = tf.unique_with_counts(valid_Y)
				sumrate = tf.reduce_sum(tf.multiply(tf.gather(self.state_rates,count_y),tf.cast(count,tf.float32)))
				s_weights.append(sumrate/tf.cast(self.s_length[b],tf.float32))
			self.fcost = tf.reduce_sum(tf.multiply(tf.multiply(-log_likelihood,s_weights),c_weights))
			#self.loss = tf.reduce_mean(-log_likelihood)
			#self.train_op = tf.train.GradientDescentOptimizer(learning_rage).minimize(self.loss)
			self.train_op = tf.train.GradientDescentOptimizer(learning_rage).minimize(self.fcost)

		else:
			self.pred,self.best_scores = tf.contrib.crf.crf_decode(self.fixed_preds,self.transition_params,self.s_length)
			
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep=10)


def build_train_corpus(batch_size,tagged_dir,source_dir):
	tagger = lstm_tagger.tagger()
	print("开始载入数据")
	states = tagger.states
	word_source,source_attr_tag,word_state_sequence,no_replaced,phrase_label = tagger.build_in_vec(tagged_dir,source_dir)
	in_vector_len = 128
	out_class_len = len(states)

	#统计各个state包含字的数量，用于平衡各个state的权重
	state_rates = [0 for i in range(out_class_len)]
	for f in word_state_sequence:
		for l in f:
			for s in l:
				state_rates[s]+=1
	#state_rates = [np.exp(s) for s in state_rates]
	state_rates = [s/sum(state_rates) for s in state_rates]
	state_rates = [-np.log(s) for s in state_rates]

	#从大到小排个序
	word_source = sorted(word_source,key=lambda l:len(l))
	source_attr_tag = sorted(source_attr_tag,key=lambda l:len(l))
	word_state_sequence = sorted(word_state_sequence,key=lambda l:len(l))
	no_replaced = sorted(no_replaced,key=lambda l:len(l))

	#按batch数分n组 快一点
	batch_iter = math.ceil(len(word_source)/batch_size)
	word_sources = []
	source_attr_tags = []
	word_state_sequences = []
	no_replaceds = []
	for f in range(batch_iter):
		word_sources.append( word_source[f*batch_size:(f+1)*batch_size])
		source_attr_tags.append( source_attr_tag[f*batch_size:(f+1)*batch_size])
		word_state_sequences.append( word_state_sequence[f*batch_size:(f+1)*batch_size])
		no_replaceds.append( no_replaced[f*batch_size:(f+1)*batch_size])

	X = []
	Y = []
	lengths = []
	true_lengths = []
	max_sentense_len = max([max([len(line) for line in f]) for f in source_attr_tag for source_attr_tag in source_attr_tags])

	for l in range(batch_iter):

		file_num = len(word_state_sequences[l])
		#每个文章有多少个时间截断（句子）
		file_num_steps = [len(f) for f in word_state_sequences[l]]
		#每个文章的每个时间截断（句子） 有多少字 在dynamicRnn中指定长度用
		#file_timesteps = [[len(line) for line in f] for f in source_attr_tags[l]]
		#文章最长句字数
		#max_sentense_len = max([max(ws) for ws in file_timesteps])

		max_file_len = max(file_num_steps)

		#print(str(l)+": maxlen"+str(max_file_len))
		#num_step = max_file_len
		#规范输入 num_step * batch * timestep * vec
		raw_X = [[] for f in range(max_file_len)]
		#规范输出 numstep * batch * statevec
		raw_Y = [[] for f in range(max_file_len)]
		#记录句中有效字数
		raw_length = [[] for f in range(max_file_len)]
		#记录当前批次 有哪些是补全的分组
		true_length = [[] for f in range(max_file_len)]

		#第几句 行内padding和句子padding
		for step_index in range(max_file_len):
			for fi in range(file_num):
				if step_index < file_num_steps[fi]:
					line_attr = copy.copy(source_attr_tags[l][fi][step_index])
					line_states = copy.copy(word_state_sequences[l][fi][step_index])
					raw_len = len(word_state_sequences[l][fi][step_index])
					true_len = raw_len
					#time_step padding
					for w in range(len(source_attr_tags[l][fi][step_index]),max_sentense_len):
						line_attr.append([0 for i in range(in_vector_len)])
						#states padding 重复最后一个状态
						line_states.append(line_states[len(line_states)-1])
					
				else :
					#尝试复制最后一句
					temp = copy.copy( raw_X[step_index-1][fi])
					line_attr =temp
					temp = copy.copy(raw_Y[step_index-1][fi])
					line_states = temp
					temp = copy.copy(raw_length[step_index-1][fi])
					raw_len = temp
					true_len = 0

				raw_X[step_index].append(line_attr)
				raw_Y[step_index].append(line_states)
				raw_length[step_index].append(raw_len)
				true_length[step_index].append(true_len)
		X.append(raw_X)
		Y.append(raw_Y)
		lengths.append(raw_length)
		true_lengths.append(true_length)
	# line_counter = 0
	# for r in raw_length:
	# 	for rr in r:
	# 		line_counter+=rr
	#print(line_counter)
	print("载入数据完毕")
	'''
	分step 的 X，Y  有效长度  不包含复制分组的真实长度  输入维度  输出维度  最长句子长度  每个词每个状态的比重
	 X: [batch_num,steps,file_len,num_step,input_dim]  
	 Y: [batch_num,steps,file_len,num_step]
	 lengths,true_lengths : [batch_num,steps,file_len]
	'''
	return X,Y,lengths,true_lengths,in_vector_len,out_class_len,max_sentense_len,state_rates,word_source

def build_corpus(batch_size,tagged_dir,source_dir):
	tagger = lstm_tagger.tagger()
	print("开始载入数据")
	states = tagger.states
	#word_source0,source_attr_tag0,word_state_sequence0,no_replaced0,phrase_label0 = tagger.build_in_vec(tagged_dir,source_dir)
	word_source,source_attr_tag,no_replaced,phrase_label = tagger.build_tag_vec(tagged_dir,source_dir)
	# if str(source_attr_tag) == str(source_attr_tag0):
	# 	print("-------")
	# else:
	# 	print("!!!!!!!!!!!!!!!!")
	in_vector_len = 128
	out_class_len = len(states)
	#从大到小排个序
	word_source = sorted(word_source,key=lambda l:len(l))
	source_attr_tag = sorted(source_attr_tag,key=lambda l:len(l))
	no_replaced = sorted(no_replaced,key=lambda l:len(l))

	#按batch数分n组 快一点
	batch_iter = math.ceil(len(word_source)/batch_size)
	word_sources = []
	source_attr_tags = []
	no_replaceds = []
	for f in range(batch_iter):
		word_sources.append( word_source[f*batch_size:(f+1)*batch_size])
		source_attr_tags.append( source_attr_tag[f*batch_size:(f+1)*batch_size])
		no_replaceds.append( no_replaced[f*batch_size:(f+1)*batch_size])

	X = []
	Y = []
	lengths = []
	true_lengths = []
	t = [[len(line) for line in f] for f in source_attr_tag for source_attr_tag in source_attr_tags]
	max_sentense_len = max(t)

	for l in range(batch_iter):

		file_num = len(source_attr_tag[l])
		#每个文章有多少个时间截断（句子）
		file_num_steps = [len(f) for f in source_attr_tag[l]]

		max_file_len = max(file_num_steps)

		#print(str(l)+": maxlen"+str(max_file_len))
		#num_step = max_file_len
		#规范输入 num_step * batch * timestep * vec
		raw_X = [[] for f in range(max_file_len)]
		#记录句中有效字数
		raw_length = [[] for f in range(max_file_len)]
		#记录当前批次 有哪些是补全的分组
		true_length = [[] for f in range(max_file_len)]

		#第几句 行内padding和句子padding
		for step_index in range(max_file_len):
			for fi in range(file_num):
				if step_index < file_num_steps[fi]:
					line_attr = copy.copy(source_attr_tags[l][fi][step_index])
					raw_len = len(source_attr_tags[l][fi][step_index])
					true_len = raw_len
					for w in range(len(source_attr_tags[l][fi][step_index]),max_sentense_len):
						line_attr.append([0 for i in range(in_vector_len)])
				else :
					temp = copy.copy( raw_X[step_index-1][fi])
					line_attr =temp
					temp = copy.copy(raw_length[step_index-1][fi])
					raw_len = temp
					true_len = 0
				raw_X[step_index].append(line_attr)
				raw_length[step_index].append(raw_len)
				true_length[step_index].append(true_len)
		X.append(raw_X)
		lengths.append(raw_length)
		true_lengths.append(true_length)
	print("载入数据完毕")
	return X,lengths,true_lengths,in_vector_len,out_class_len,max_sentense_len,word_source

def train_lstm_crf(batch_size,X,Y,lengths,true_lengths,in_vector_len,out_class_len,state_rates):

	#batch_size = 10
	training_iters = 10000
	model = LSTM_CRF(batch_size,in_vector_len,out_class_len,True)
	model.build_model()
	print("train start")
	#加载hmm相关参数
	_transition_params = np.loadtxt("HMM_transfer.txt")
	with tf.Session() as session:
		#第一次训练
		#session.run(model.init)
		#接着上一个保存点继续训练
		fdir = "lstm_models"
		ckpt = tf.train.get_checkpoint_state(fdir)
		if ckpt and ckpt.model_checkpoint_path:
			model.saver.restore(session, ckpt.model_checkpoint_path)  
		iter_num = 360
		#_transition_params = _transfer
		while iter_num < training_iters:
			print("------"+str(iter_num)+"-------")
			start = time.time()
			state = None
			saved_accuracy = []
			maincost = 0
			#检验每个类别的正确率
			state_accurate_words = [0 for i in range(out_class_len)]
			all_state_words = [0 for i in range(out_class_len)]
			for bi in range(len(X)):
				raw_X = X[bi]
				raw_Y = Y[bi]
				raw_length = lengths[bi]
				true_length = true_lengths[bi]
				for step in range(len(raw_X)):
					x = raw_X[step]
					y = raw_Y[step]
					#每个batch的上一次输出的最后一个字的转移概率 作为本次运算的初始概率 [batch_size,out_class_len]
					if step == 0:
						_prestep_laststate =  [out_class_len-1 for f in range(model.batch_size)] 
						_pred,state,_transition_params = session.run([model.pred,model.cell_final_state,model.transition_params],feed_dict={
							model.X:x,
							model.s_length:raw_length[step],
							model.prestep_laststate : _prestep_laststate,
							model.transition_params : _transition_params
						})
					else:
						# a0 = np.asarray(x)
						# print(str(bi)+" - "+str(step)+" : "+str(a0.shape))
						_prestep_laststate = []
						for i in range(model.batch_size):
							_prestep_laststate.append(_pred[i][raw_length[step-1][i]-1])

						_pred,state,_transition_params = session.run([model.pred,model.cell_final_state,model.transition_params],feed_dict={
							model.X:x,
							model.s_length:raw_length[step],
							model.hidden_state : state.h,
							model.current_state : state.c,
							model.prestep_laststate : _prestep_laststate,
							model.transition_params : _transition_params
						})

					_,_costs = session.run([model.train_op,model.fcost],feed_dict={
							model.X: x,
							model.Y:y,
							model.s_length:raw_length[step],
							model.t_length:true_length[step],
							model.pred:_pred,
							model.prestep_laststate : _prestep_laststate,
							model.transition_params : _transition_params,
							model.state_rates : state_rates
						})
					
					maincost+=_costs
					
					#计算准确率
					for b in range(model.batch_size):
						for i in range(raw_length[step][b]):
							all_state_words[y[b][i]] +=1
							if _pred[b][i] == y[b][i]:
								state_accurate_words[y[b][i]]+=1

			state_accurate = []

			for i,ii in enumerate(all_state_words):
				if ii==0:
					state_accurate.append(-1)
				else:
					state_accurate .append( state_accurate_words[i]/all_state_words[i])

			end = time.time()
			print("time : "+ str(end-start))
			#print("average:"+np.mean(accuracys))
			#print("min :"+str(min(saved_accuracy)) + " position:"+str(saved_accuracy.index(min(saved_accuracy))))
			print(state_accurate)
			print(state_accurate_words)
			print(all_state_words)
			print("main cost :"+str(maincost))
			iter_num+=1

			r_file = codecs.open("lstm_models/statis.txt",'a+','utf-8')
			r_file.write(str(state_accurate)+" "+str(maincost) +"\n")
			r_file.close()

			np.savetxt("HMM_transfer.txt", np.array(_transition_params));

			if iter_num%5 == 0:
				model.saver.save(session,"lstm_models/lstm_model.ckpt",global_step=iter_num)
				#saver.restore(session,"lstm_model.ckpt")
				print("save model over")

def exam_lstm_crf():
	#tagger = lstm_tagger.tagger()
	#states = tagger.states
	_transition_params = np.loadtxt("HMM_transfer.txt")
	tagged_dir = "test_set"
	#tagged_dir = "test_corpus_less"
	#tagged_dir = "test_hmm"
	source_dir = "tagged"
	file_num = len(os.listdir(tagged_dir))
	batch_size = 10
	#word_source,attr_tag_source,word_state_sequence,no_replaced,phrase_labels = tagger.build_in_vec(tagged_dir,source_dir)
	X,Y,lengths,true_lengths,in_vector_len,out_class_len,maxlens,state_rates,word_source = build_train_corpus(batch_size,tagged_dir,source_dir)
	
	model = LSTM_CRF(batch_size,in_vector_len,out_class_len,False)
	model.build_model()

	#fbi step bi 
	all_pred = [[[[] for b in range(batch_size)] for s in x] for x in X]
	true_label = [[[[] for b in range(batch_size)] for s in x] for x in X]

	with tf.Session() as session:
		fdir = "lstm_models"
		ckpt = tf.train.get_checkpoint_state(fdir)
		print("*"*10)
		if ckpt and ckpt.model_checkpoint_path:
			model.saver.restore(session, ckpt.model_checkpoint_path)
		for bi in range(len(X)):
			raw_X = X[bi]
			raw_Y = Y[bi]
			raw_length = lengths[bi]
			true_length = true_lengths[bi]
			for step in range(len(raw_X)):
				x = raw_X[step]
				y = raw_Y[step]

				#每个batch的上一次输出的最后一个字的转移概率 作为本次运算的初始概率 [batch_size,out_class_len]
				if step == 0:
					_prestep_laststate =  [12 for f in range(model.batch_size)] 
					_pred,state,_transition_params = session.run([model.pred,model.cell_final_state,model.transition_params],feed_dict={
						model.X:x,
						model.s_length:raw_length[step],
						model.prestep_laststate : _prestep_laststate,
						model.transition_params : _transition_params
					})
				else:
					_prestep_laststate = []
					for i in range(model.batch_size):
						_prestep_laststate.append(_pred[i][raw_length[step-1][i]-1])
					_pred,state,_transition_params = session.run([model.pred,model.cell_final_state,model.transition_params],feed_dict={
						model.X:x,
						model.s_length:raw_length[step],
						model.hidden_state : state.h,
						model.current_state : state.c,
						model.prestep_laststate : _prestep_laststate,
						model.transition_params : _transition_params
					})

				################校验开始#################
				for batch_i,pred in enumerate(_pred):
					for si in range(true_length[step][batch_i]):
						all_pred[bi][step][batch_i].append(pred[si])
						true_label[bi][step][batch_i].append(y[batch_i][si])
		

		count_accuracy(all_pred,true_label,word_source,batch_size,out_class_len,tagged_dir)

def count_accuracy(all_pred,true_label,word_source,batch_size,out_class_len,tagged_dir):
	tagfiles = os.listdir(tagged_dir)
	file_num = len(tagfiles)
	#判断正确
	correct = [[0 for j in range(out_class_len)] for i in range(file_num)]
	#对的判成错的
	wrong = [[0 for j in range(out_class_len)] for i in range(file_num)]
	fin_result = [["" for i in range(out_class_len)] for j in range(file_num)]

	#统计一下每个标注的个数
	tag_total = [[0 for s in range(out_class_len)] for f in tagfiles]
	stotal = [[0 for s in range(out_class_len)] for f in tagfiles]
	for bi in range(len(all_pred)):
		for step in range(len(all_pred[bi])):
			for b in range(len(all_pred[bi][step])):
				for si,s in enumerate( all_pred[bi][step][b]):
					tag_total[b+batch_size*bi][s] += 1
					tag_s = true_label[bi][step][b][si]
					stotal[b+batch_size*bi][tag_s] += 1

	for bi in range(len(all_pred)):
		for step in range(len(all_pred[bi])):
			for b in range(len(all_pred[bi][step])):
				if tag_total[b+batch_size*bi][11]<5:
					for si,s in enumerate( all_pred[bi][step][b]):
						if s==11:
							all_pred[bi][step][b][si]=12


	#成果部分 如果字数少于5 那就全被定为无关文本

	for bi in range(len(all_pred)):
		for step in range(len(all_pred[bi])):
			for b in range(len(all_pred[bi][step])):
				for si,s in enumerate( all_pred[bi][step][b]):
					tag_s = true_label[bi][step][b][si]
					if tag_s == s:
						correct[b+batch_size*bi][s] += 1
					else:
						#此位置 错判  错判成的那个分类 +=1
						wrong[b+batch_size*bi][s] += 1
					#根据状态 添加到不同的结果中取
					fin_result[b+batch_size*bi][s]+=word_source[b+batch_size*bi][step][si]+" "

	for f_index in range(file_num):
		t_file=codecs.open("test_result/"+str(f_index)+".txt",'w', encoding='utf8')
		t_file.write("\n".join(fin_result[f_index]))
		t_file.close()


	precisions = [[0 for j in  range(out_class_len)] for i in range(file_num)] 
	recalls = [[0 for j in  range(out_class_len)] for i in range(file_num)]
	for fi,f in enumerate(precisions):
		for si,num in enumerate(f):
			if correct[fi][si] + wrong[fi][si] == 0 :
				precisions[fi][si] = 1
			else:
				precisions[fi][si] = correct[fi][si]/(correct[fi][si] + wrong[fi][si])
			if stotal[fi][si] == 0:
				recalls[fi][si] = 1
			else:
				recalls[fi][si] = correct[fi][si] / stotal[fi][si]

	s_precisions = [0 for i in  range(out_class_len)]
	s_recalls = [0 for i in  range(out_class_len)]


	for si in range(out_class_len):

		if si in [1,3,5,7,9,11]:
			print("-----------"+str(si)+"-----------")
			t = []
			for f in precisions:
				t.append(f[si])
			precision = sum(t)/len(t)
			print("  precision  "+ str(precision))
			# 看看最低值的情况
			# if si == 11:
				
			# 	prefdic = {}
			# 	for ti in range(len(t)):
			# 		prefdic[tagfiles[ti]] = str(t[ti])+"  "+str(ti)
			# 	finfdic = sorted(prefdic.items(),key=lambda item:item[1])
			# 	print(finfdic[0:40])
			# #print(str(sum(t)/len(t)))
			t = []
			for f in recalls:
				t.append(f[si])
			recall = sum(t)/len(t)
			print("  average  "+ str(recall))
			F_value = precision*recall*2/(precision+recall)
			print("  F  "+ str(F_value))


batch_size = 10
tags_dir = "train_set"
train_dir = "train_set"
#tagged_dir = "test_corpus_less"
source_dir = "tagged"
# x,y,length,true_lengths,in_vector_len,out_class_len,maxlens,state_rates,_ = build_train_corpus(batch_size,train_dir,source_dir)
# # print(state_rates)
# print("start dump")
# pickle.dump(x, open("raw_X.txt", 'wb'))
# print("load x over")
# pickle.dump(y, open("raw_Y.txt", 'wb'))
# print("load y over")
# pickle.dump(length, open("raw_length.txt", 'wb'))
# pickle.dump(true_lengths, open("true_lengths.txt", 'wb'))
# pickle.dump(state_rates, open("state_rates.txt", 'wb'))

#服务器内训练
# in_vector_len = 128
# out_class_len = 13
# with open('raw_X.txt','rb') as raw_X:
# 	x = pickle.load(raw_X)
# print("load x over")
# with open('raw_Y.txt','rb') as raw_Y:
# 	y = pickle.load(raw_Y)
# print("load Y over")
# with open('raw_length.txt','rb') as raw_length:
# 	length = pickle.load(raw_length)
# print("load raw_length over")
# with open('state_rates.txt','rb') as raw_state_rates:
# 	state_rates = pickle.load(raw_state_rates)
# print("load maxlens over")
# with open('true_lengths.txt','rb') as raw_true_lengths:
# 	true_lengths = pickle.load(raw_true_lengths)
# print("load true_lengths over")


#train_lstm_crf(batch_size,x,y,length,true_lengths,in_vector_len,out_class_len,state_rates)
exam_lstm_crf()
