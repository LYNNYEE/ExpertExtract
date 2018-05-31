import codecs
import jieba
import jieba.posseg as pseg
from jpype import *
import re 
from bs4 import BeautifulSoup
import os
from gensim.models import Word2Vec
import operator
import shutil

class tagger():

	states = ["name_d","name","title_d","title","email_d","email","search_d","search","experience_d","experience","product_d","product","others"]
	wordvec_len = 128
	labels = ["name_label","place_label","maillabel","phonelabel","timelabel","patentlabel","projectlabel","booklabel"]

	def __init__(self):
		startJVM(getDefaultJVMPath(),"-Djava.class.path=D:/HanLP/hanlp-1.3.4.jar;D:/HanLP", "-Xms1g", "-Xmx1g")
		HanLp = JClass('com.hankcs.hanlp.HanLP')
		self.segment = HanLp.newSegment().enableOrganizationRecognize(True)
		jieba.load_userdict("shuyusum_new.txt")
		self.model = Word2Vec.load("teacher_model")

	#去除影响分段的div标志
	def fomart_html(self,raw_html):

		raw_html = re.sub(r'<span[^>]*>','',raw_html)
		raw_html= re.sub(r'</span>','',raw_html)

		raw_html = re.sub(r'<font[^>]*>','',raw_html)
		raw_html= re.sub(r'</font>','',raw_html)

		raw_html = re.sub(r'<a[^>]*>','',raw_html)
		raw_html= re.sub(r'</a>','',raw_html)

		raw_html = re.sub(r'<label[^>]*>','',raw_html)
		raw_html= re.sub(r'</label>','',raw_html)

		raw_html = re.sub(r'<strong>','',raw_html)
		raw_html = re.sub(r'</strong>','',raw_html)

		raw_html = re.sub(r'<em>','',raw_html)
		raw_html = re.sub(r'</em>','',raw_html)

		raw_html= re.sub(r'<b>','',raw_html)
		raw_html= re.sub(r'</b>','',raw_html)

		raw_html= re.sub(r'<u>','',raw_html)
		raw_html= re.sub(r'</u>','',raw_html)

		#raw_html= re.sub('\\xa0','',raw_html)

		soup = BeautifulSoup(raw_html,"lxml")

		for t in soup.find_all('script'):
			t.decompose() 
		for t in soup.find_all('style'):
			t.decompose() 
		for t in soup.find_all('br'):
			t.decompose() 

		res = []
		if soup.body is None:
			return []
		for text in soup.body.stripped_strings:
			#text = re.sub(" ","",text)
			text = re.sub("\\xa0","",text)
			text = re.sub('[\t\r\n]+', ' ', text)
			#text = re.sub('\\u3000','',text)
			text=text.strip()
			if len(text)>0:
				#res.append(text)
				res.append(text)
			#print(tlist)

		return res


	#提前抽取替换词表 
	def format_replace_dic(self,res):
		patterns = {}
		#匹配邮箱
		patterns['maillabel'] = r"([\w!#$%&'*+/=?^_`{|}~-]+(?:\.[\w!#$%&'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?)"
		#匹配电话号码
		patterns['phonelabel'] = r"(\d{3}-\d{8}|\d{4}-\{7,8})"
		#匹配时间
		patterns['timelabel'] = r"((19\d\d|2[01]\d\d年?)([^0-9]?(\d|[01]\d)(?=\D)月?)?([^0-9]?(\d|[0-3]\d)(?=\D)日?)?)"
		#匹配专利号
		patterns['patentlabel'] = r"([A-Z]{2}\d+)"
		#匹配项目号
		patterns['projectlabel'] = r"(\d{4}[A-Z]{2}[A-Za-z0-9]{6})"
		#匹配书名号内容
		patterns['booklabel'] = r"(《[^《^》]+》)"
		#匹配奇怪的号码 默认为成果编号 暂定
		#numsPattern = r""

		replaced_words = {}
		for line in res:
			for key in patterns.keys():
				pattern = patterns[key]
				mat_words = re.findall(pattern,line)
				for word in mat_words:
					if isinstance(word,tuple):
						word = word[0]
					temp = replaced_words.get(key,[])
					temp.append(word)
					replaced_words[key] = temp

			#人名识别使用Hanlp的 MEMM 方法
			segs = self.segment.seg(line)
			for s in segs:
				if str(s.nature)=="nr" or str(s.nature)=="nrf":
					temp = replaced_words.get("name_label",[])
					temp.append(s.word)
					replaced_words['name_label'] = temp
				# if str(s.nature)=="ns":
				# 	temp = replaced_words.get("place_label",[])
				# 	temp.append(s.word)
				# 	replaced_words['place_label'] = temp

		return replaced_words

	#替代掉指定词语 返回分词后的结果 和 替换的位置
	def get_W2V_train_corpse(self,res,replaced_words):
		vec_train_words = []
		#replacewords_positions = []
		#res = self.fomart_html(source_texts)
		#添加替代词库
		for key in replaced_words:
			for w in replaced_words[key]:
				jieba.add_word(w)
		#记录被替换的词，便于还原
		rep_ws = []
		for line_i,line in enumerate(res):
			sentence = []
			ws = jieba.cut(line)
			for w_i,w in enumerate(ws):
				fw = w
				for key in replaced_words:
					if w in replaced_words[key]:
						fw = key
						rep_ws.append(w)
						#replacewords_positions.append((line_i,w_i))
						break
				sentence.append(fw)
			vec_train_words.append(sentence)
		return vec_train_words,rep_ws

	#从人工标注中获取没个词的状态
	def read_tag_docs(self,source_texts,tag_texts):

		attr_tag_source = [[12 for j in i] for i in source_texts]
		word_state_sequence = [[] for i in source_texts]

		index_tag_pos = tag_texts.split("\n")

		for index, tag in enumerate(index_tag_pos):
			if len(tag) == 0:
				continue

			spaces = tag.split(",")
			#space:[14-47:[]]/[14:[]]/14:[0:6]
			for space in spaces:
				iw = space.split("/")
				if len(iw) <2:
					continue
				test_word = []
				test_word_tag = []
				#如果开头包含- 那么肯定包含该块所有内容
				if "-" in iw[0]:
					ii = iw[0].split("-")
					i0 = int(ii[0])-1
					i1 = int(ii[1])
					for i in range(i0,i1):
						for t in range(len(source_texts[i])):
							if attr_tag_source[i][t] != 12:
								print("tag err"+str(i)+" "+str(t))
							#在原attr_tag中标记所属类别 未被标记的都是上下文
							attr_tag_source[i][t] = index

				else:
					source_line = source_texts[int(iw[0])-1]
					tag_line = attr_tag_source[int(iw[0])-1]
					if "[]" in iw[1] :
						for i in range(len(source_line)):
							if tag_line[i] != 12:
								print("tag err"+str(int(iw[0])-1)+" "+str(i))
							tag_line[i] = index
					else:
						temp_list = eval("source_line"+iw[1])
						for i in range(len(temp_list)):
							#标注中的位置
							tag_ind =  eval("[i for i in range(len(source_line))]"+iw[1])[i]
							if tag_line[tag_ind] != 12:
								print("tag err"+str(int(iw[0])-1)+" "+str(i))
							#test_strs_tags .append(str(tag_line[tag_ind]))
							tag_line[tag_ind] = index

		return attr_tag_source

	#将该标注风格转移到别的风格中去
	def transfer_tag_docs(self,source_texts,tag_texts):
		#attr_tag_source = [[12 for j in i] for i in source_texts]
		#word_state_sequence = [[] for i in source_texts]
		tag_texts = re.sub("\r","",tag_texts)
		index_tag_pos = tag_texts.split("\n")
		transfer_sources = ""
		for index, tag in enumerate(index_tag_pos):
			
			if len(tag)==0 :
				transfer_sources +="\n"
				continue
			spaces = tag.split(",")
			#space:[14-47:[]]/[14:[]]/14:[0:6]
			for space in spaces:
				iw = space.split("/")
				if len(iw) <2:
					continue
				#如果开头包含- 那么直接复制
				if "-" in iw[0]:
					transfer_sources+=space
				else:
					source_line = source_texts[int(iw[0])-1]
					if "[]" in iw[1] :
						transfer_sources+=space
					else:
						transfer_sources+=iw[0]+"/"+"|"+str(eval("source_line"+iw[1]))
				transfer_sources+=","
			transfer_sources = transfer_sources[:-1]
			transfer_sources +="\n"
		return transfer_sources


	#删除原文本中的标点符号 及开头的1234， #并记录 哪些符号内是一句话 计算时按句计算 (错误的)
	def del_symbols(self,source_texts):
		del_position = []
		phrase_labels = [[] for i in source_texts]
		nosymbol_source = [[] for i in source_texts]
		pattern = r"[^\u4e00-\u9fa5a-zA-Z0-9]+"
		number_headword = [str(i) for i in range(50)]
		number_headword.extend(["一","二","三","四","五","六","七","八","九"])
		for i,line in enumerate(source_texts):
			line_phrase_counter = 0
			number_headflag = False
			for j,w in enumerate(line):
				if re.match(pattern,w):
					line_phrase_counter+=1
					del_position.append((i,j))
				else:
					if j ==0:
						if w in number_headword:
							line_phrase_counter+=1
							del_position.append((i,j))
							continue
					nosymbol_source[i].append(w)
					phrase_labels[i].append(line_phrase_counter)
		return nosymbol_source,phrase_labels,del_position

	#根据所有的文件 构建word2vec模型
	def build_word2vec_model(self):
		doc = []
		source_dir = "D:/expertSearch/college/college/htmls"
		nosymbol_source_dir = "D:/expertSearch/college/college/seged_htmls"
		u_dir = os.listdir(source_dir)
		counter = 0
		for di, d in enumerate(u_dir):
			if d =="error.txt":
				continue
			fs = os.listdir(source_dir+"/"+d)
			for f in fs:
				if f == "log.txt":
					continue
				file_object = codecs.open(source_dir+"/"+d+"/"+f , "r", "utf-8")
				source_texts = file_object.read()
				file_object.close()
				# if source_texts is None:
				# 	continue
				# if len(source_texts) == 0:
				# 	continue
				res = self.fomart_html(source_texts)
				# if len(res) == 0:
				# 	continue
				replaced_words = self.format_replace_dic(res)
				source_texts,rep_ws = self.get_W2V_train_corpse(res,replaced_words)
				nosymbol_source,phrase_labels,del_position = self.del_symbols(source_texts)
				for t in  nosymbol_source:
					if len(t)>0:
						doc.append(t)

				#把分词文件保存
				saved_str = "\n".join([" ".join(i) for i in nosymbol_source])
				if not os.path.exists(nosymbol_source_dir+"/"+d):
					os.makedirs(nosymbol_source_dir+"/"+d)
				t_dir = nosymbol_source_dir+"/"+d+"/"+f
				t_file=codecs.open(t_dir,'w', encoding='utf8')
				t_file.write(saved_str)
				t_file.close()

				counter+=1
				print(counter)
		model = Word2Vec(doc,size=self.wordvec_len,min_count=1)
		model.save('teacher_model')
		print("---training model---")

	#把尚未标记的文件给筛选出来给师弟用
	def not_labeled_files(self):
		u_dir = "D:/expertSearch/college/tagged"
		t_dir = "D:/expertSearch/college/tagged_tag"
		shidixinkule = "D:/expertSearch/college/shidixinkule"
		u_dirs = os.listdir(u_dir)
		t_dirs = os.listdir(t_dir)
		for u in u_dirs:
			if "tag_" not in u:
				if "tag_"+u not in t_dirs:
					shutil.copyfile(u_dir+"/"+u,shidixinkule+"/"+u)

	#测试人工标记是否正确
	def test_humanlabel(self,fname,tag_fname):
		file_object = codecs.open(fname , "r", "utf-8")
		source_texts = file_object.read()
		file_object.close()
		file_object = codecs.open(tag_fname , "r", "utf-8")
		tag_positions = file_object.read()
		file_object.close()
		res = self.fomart_html(source_texts)
		replaced_words = self.format_replace_dic(res)
		source_texts,rep_ws = self.get_W2V_train_corpse(res,replaced_words)
		s = self.read_tag_docs(source_texts,tag_positions)
		nosymbol_source,phrase_label,del_position = self.del_symbols(source_texts)
		res = ["" for i in self.states]
		ress = [[] for i in self.states]
		for linei, line in enumerate (s):
			for wsi ,ws in enumerate (line):
				res[ws]+=source_texts[linei][wsi]
				ress[ws].append(source_texts[linei][wsi])
		#for r in res:
			# print(r)
			# print("-"*10)
		return ress,source_texts
	
	def test_humanlabel1(self,source_dir,tag_dir):
		word_source,source_attr_tag,word_state_sequence,no_replaceds,phrase_labels = self.build_in_vec(tag_dir,source_dir)
		res = ["" for i in self.states]
		ress = [[] for i in self.states]
		for fi,f in enumerate(word_source):
			for linei, line in enumerate (word_state_sequence[fi]):
				for wsi ,ws in enumerate (line):
					res[ws]+=word_source[fi][linei][wsi]
					ress[ws].append(word_source[fi][linei][wsi])
		
		for li,l in enumerate(ress):
			
			print("------"+str(li)+"-------")
			with codecs.open("exam_train_set"+"/"+str(li)+".txt",'w','utf-8') as f:
				f.write(str(l))

		#return ress,source_texts
	# def test_humanlabel(self,tagged_dir,source_dir):
	# 	word_source,source_attr_tag,word_state_sequence,no_replaceds,phrase_labels = self.build_in_vec(tagged_dir,source_dir)
	# 	hum_result = ["" for i in range(13)]
	# 	for f_index in range(len(source_attr_tag)):
	# 		for line_index,line in enumerate(word_state_sequence[f_index]):
	# 			for si,s in enumerate(line):
	# 				hum_result[int(s)] += no_replaced[f_index][line_index][si]+" "
	# 		for i in range(13):
	# 			print(hum_result[i])
	# 			print("-"*5+str(f_index)+"-"*5)
	# 		print("*"*10)


	#调用word2vec模型 获得指定词的词向
	def get_wordvec(self,word):
		return self.model.wv[word]

	def build_tag_vec(self,tagged_dir,source_dir):
		f_list = os.listdir(source_dir)
		word_source = [[] for i in range(len(f_list))]
		source_attr_tag = [[] for i in range(len(f_list))]
		no_replaceds = [[] for i in range(len(f_list))]
		phrase_labels = [[] for i in range(len(f_list))]

		f_list = os.listdir(tagged_dir)
		
		for f_index,f in enumerate(f_list):
			path = source_dir+"\\"+f[4:]
			file_object = codecs.open(path , "r", "utf-8")
			source_texts = file_object.read()
			file_object.close()
			res = self.fomart_html(source_texts)
			replaced_words = self.format_replace_dic(res)
			source_texts,rep_ws = self.get_W2V_train_corpse(res,replaced_words)
			nosymbol_source,phrase_label,del_position = self.del_symbols(source_texts)
			#还原被替换的元素
			no_replaced = [[] for i in nosymbol_source]
			rep_counter = 0
			for line_i, line in  enumerate(nosymbol_source):
				for w_i, w in enumerate(line):
					if w in self.labels:
						no_replaced[line_i].append(rep_ws[rep_counter])
						rep_counter+=1
					else:
						no_replaced[line_i].append(w)

			no_replaced = [i for i in no_replaced if len(i) > 0]

			#去无内容项 保存有内容项
			nosymbol_source = [i for i in nosymbol_source if len(i)>0]
			#nosymbol_tags = [i for i in nosymbol_tags if len(i)>0]
			nosymbol_source_vec = [[[] for j in i] for i in nosymbol_source]
			phrase_label = [i for i in phrase_label if len(i)>0]
			#填充向量，避免oov
			for line_i,line in enumerate(nosymbol_source):
				for w_i,w in enumerate(line):
					try:
						v = self.get_wordvec(w)
					except Exception as err:
						v = self.model.wv["的"]
					nosymbol_source_vec[line_i][w_i] = v

			word_source[f_index] = nosymbol_source
			#word_state_sequence[f_index] = nosymbol_tags
			source_attr_tag[f_index] = nosymbol_source_vec
			no_replaceds[f_index] = no_replaced
			phrase_labels[f_index] = phrase_label

		#替换过的词序列  替换过的词向量序列 未替换的词序列  每行的每个词属于哪个句子
		return word_source,source_attr_tag,no_replaceds,phrase_labels


	#构建lstm输入
	def build_in_vec(self,tagged_dir,source_dir):

		#获得原始内容
		f_list = os.listdir(tagged_dir) #列出文件夹下所有的目录与文件

		#记录每个词的本身，状态和词向量
		word_source = [[] for i in range(len(f_list))]
		word_state_sequence = [[] for i in range(len(f_list))]
		source_attr_tag = [[] for i in range(len(f_list))]
		no_replaceds = [[] for i in range(len(f_list))]
		phrase_labels = [[] for i in range(len(f_list))]
		#人工检查标注情况
		#test = ["" for i in self.states]

		for f_index,f in enumerate(f_list):
			#print(f_list[f_index])
			path = os.path.join(tagged_dir,f)
			file_object = codecs.open(path , "r", "utf-8")
			tag_positions = file_object.read()
			file_object.close()
			
			path = source_dir+"\\"+f[4:]
			file_object = codecs.open(path , "r", "utf-8")
			source_texts = file_object.read()
			file_object.close()

			res = self.fomart_html(source_texts)
			replaced_words = self.format_replace_dic(res)
			source_texts,rep_ws = self.get_W2V_train_corpse(res,replaced_words)
			source_tags = self.read_tag_docs(source_texts,tag_positions)
			nosymbol_source,phrase_label,del_position = self.del_symbols(source_texts)

			#还原被替换的元素
			no_replaced = [[] for i in nosymbol_source]
			rep_counter = 0
			for line_i, line in  enumerate(nosymbol_source):
				for w_i, w in enumerate(line):
					if w in self.labels:
						no_replaced[line_i].append(rep_ws[rep_counter])
						rep_counter+=1
					else:
						no_replaced[line_i].append(w)

			no_replaced = [i for i in no_replaced if len(i) > 0]

			nosymbol_tags = [[] for i in nosymbol_source]

			for line_i,line in enumerate(source_tags):
				for s_i ,s in enumerate(line):
					#如果不在删除标点的记录位置
					if not max([operator.eq((line_i,s_i),temp) for temp in del_position]):
						nosymbol_tags[line_i].append(s)

			#去无内容项 保存有内容项
			nosymbol_source = [i for i in nosymbol_source if len(i)>0]
			nosymbol_tags = [i for i in nosymbol_tags if len(i)>0]
			nosymbol_source_vec = [[[] for j in i] for i in nosymbol_source]
			phrase_label = [i for i in phrase_label if len(i)>0]
			#填充向量，避免oov
			for line_i,line in enumerate(nosymbol_source):
				for w_i,w in enumerate(line):
					try:
						v = self.get_wordvec(w)
					except Exception as err:
						v = self.model.wv["的"]
					nosymbol_source_vec[line_i][w_i] = v

			#人工检查标注情况
			# test = ["" for i in range(13)]
			# for line_i,line in enumerate(nosymbol_tags):
			# 	for s_i,s in enumerate(line):
			# 		test[s] += nosymbol_source[line_i][s_i]

			# t_file = codecs.open("hum_exam1/" + f_list[f_index] , 'w', encoding='utf8')
			# t_file.write("\n".join(test))
			# t_file.close()

			word_source[f_index] = nosymbol_source
			word_state_sequence[f_index] = nosymbol_tags
			source_attr_tag[f_index] = nosymbol_source_vec
			no_replaceds[f_index] = no_replaced
			phrase_labels[f_index] = phrase_label

		#替换过的词序列  替换过的词向量序列  状态序列  未替换的词序列  每行的每个词属于哪个句子
		return word_source,source_attr_tag,word_state_sequence,no_replaceds,phrase_labels

	#创建文件 便于人工标签 即 把所有特殊文本替换成label之后的结果
	def create_prelabel_file(self):
		pre_dir = "tagged"
		files = os.listdir(pre_dir)
		for f in files:
			if "tag_" not in f:
				file_object = codecs.open(pre_dir+"/"+f , "r", "utf-8")
				source = file_object.read()
				file_object.close()

				res = self.fomart_html(source)
				replaced_words = self.format_replace_dic(res)
				source_texts,rep_ws = self.get_W2V_train_corpse(res,replaced_words)

				fstr = ""
				for l in source_texts:
					for wi,w in enumerate(l):
						#fstr += w+"["+str(wi)+"]"+" "
						fstr += w+" "
					fstr+="\n"

				t_file=codecs.open(pre_dir+"/tag_"+f,'w', encoding='utf8')
				#t_file.write("["+"]\n[".join([" ".join(i) for i in source_texts])+"]")
				t_file.write(fstr)
				t_file.close()


	#给定一个文件 给出内容 和 内容对应的向量
	def build_wordandvec(self,fpath):
		file_object = codecs.open(fpath , "r", "utf-8")
		source = file_object.read()
		file_object.close()

		res = self.fomart_html(source)
		replaced_words = self.format_replace_dic(res)
		source_texts,rep_ws = self.get_W2V_train_corpse(res,replaced_words)
		nosymbol_source,phrase_label,del_position = self.del_symbols(source_texts)
		nosymbol_source = [i for i in nosymbol_source if len(i)>0]

		#还原被替换的元素
		no_replaced = [[] for i in nosymbol_source]
		rep_counter = 0
		for line_i, line in  enumerate(nosymbol_source):
			for w_i, w in enumerate(line):
				if w in self.labels:
					no_replaced[line_i].append(rep_ws[rep_counter])
					rep_counter+=1
				else:
					no_replaced[line_i].append(w)

		no_replaced = [i for i in no_replaced if len(i) > 0]

		nosymbol_source_vec = [[[] for j in i] for i in nosymbol_source]
		for line_i,line in enumerate(nosymbol_source):
				for w_i,w in enumerate(line):
					try:
						v = self.get_wordvec(w)
					except Exception as err:
						print(w +" oov" + "   " )
						# self.model.build_vocab([w],update=True)
						# self.model.train([w],total_examples=self.model.corpus_count, epochs=self.model.iter)
						v = self.model.wv["的"]
					nosymbol_source_vec[line_i][w_i] = v

		return nosymbol_source,nosymbol_source_vec,no_replaced

# import shutil
tagger = tagger()
# fdir = "D:/expertSearch/college/tagged"
# tdir = "D:/expertSearch/college/last"
# tagger.test_humanlabel(tdir,fdir)
# tagger.build_in_vec(tdir,fdir)
# tagger.not_labeled_files()
# tagger.create_prelabel_file()
# tagger.build_word2vec_model()
# # # # #tagger.build_word2vec_model()
# # # tagger.build_in_vec()
#tagger.create_prelabel_file()
# # fname = "0b14def8-d398-11e7-ad8d-005056c00008.txt"
# # fdir = "D:/expertSearch/college/西安交通大学"test_corpus
fdir = "D:/expertSearch/college/tagged"
tdir = "D:/expertSearch/college/train_set"
tagger.test_humanlabel1(fdir,tdir)

# flist = os.listdir(tdir)
# tres = [[] for i in tagger.states]
# for f in flist:
# 	tf = f.split("_")[1]
# 	print(f)
# 	res,s = tagger.test_humanlabel(fdir+"/"+tf,tdir+"/"+f)
# 	for ri,r in enumerate(res):
# 		tres[ri].append(r)
# 	print("--")

# v = [0,1,2]
# for ti,t in enumerate(tres):
# 	if ti in v:
# 		print("---"+str(ti)+"---")
# 		print(t)
# 	# if ti ==8:
# 	# 	print(t)
# 	# 	for tti,tt in enumerate(t):
# 	# 		if "中央财经大学" in tt or "马克思主义理论" in tt:
# 	# 			print("****")
# 	# 			print(flist[tti])
# print("--")
# # #



###5.25 转换标注###
# tagger = tagger()
# fdir = "D:/expertSearch/college/tagged"
# tdir = "D:/expertSearch/college/train_set"
# trans_dir = "D:/expertSearch/college/new_train_set"
# flist = os.listdir(tdir)
# for fi,f in enumerate(flist):

# 	file_object = codecs.open(fdir+"/"+f.split("_")[1] , "r", "utf-8")
# 	source_texts = file_object.read()
# 	file_object.close()
# 	file_object = codecs.open(tdir+"/"+f , "r", "utf-8")
# 	tag_texts = file_object.read()
# 	file_object.close()

# 	res = tagger.fomart_html(source_texts)
# 	replaced_words = tagger.format_replace_dic(res)
# 	source_texts,rep_ws = tagger.get_W2V_train_corpse(res,replaced_words)
# 	#sources = [line.split(" ") for line in source_texts.split("\n")]
# 	if f =="tag_e11210e8-d3a1-11e7-abe6-005056c00008.txt":
# 		print("ok")

# 	transfer_text = tagger.transfer_tag_docs(source_texts,tag_texts)
# 	filename = 'write_data.txt'
# 	with codecs.open(trans_dir+"/"+f,'w','utf-8') as f:
# 		f.write(transfer_text)