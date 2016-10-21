import pandas as pd
from pandas import DataFrame
import json
import argparse

def read_testset(jsonfile):
	with open(jsonfile) as json_data:
		json_data = json.load(json_data)
	#json_data = [{'id': 10, 'tags': "hurry, cane, mail, red, blue"}, {'id': 11, 'tags': "red, elegant, stuff, came, nothing"}]
	df = DataFrame(json_data)
	return df;

def get_vocset(df_testset):
	l = len(df_testset)
	voc_set = set()
	for i in range(0,l):
		tags = df_testset.loc[i]['tags'];
		#tags = [x.strip() for x in tags.split(',')];
		voc_set = voc_set.union(set(tags));
	return voc_set;

def build_rev_word_index(df, voc_set):
	item_lists = list()
	voc_list = list(voc_set)
	for tag in voc_list:
		l = len(df)
		tmp_list = list()
		for i in range(0,l):
			tags = df.loc[i]['tags']
			#print(set([x.strip() for x in tags.split(',')]))
			if tag in set(tags): #set([x.strip() for x in tags.split(',')]):
				tmp_list.append(str(df.loc[i]['id']))
		item_lists.append(tmp_list)
	df_rev = DataFrame()
	df_rev = df_rev.assign(tag = voc_list)
	df_rev = df_rev.assign(item_lists = item_lists)
	return df_rev

# gt_jsonfile = "gt.json"
# ht_jsonfile = "ht.json"
def eval(gt_jsonfile, ht_jsonfile):
	#read groundtruth
	df_gt = read_testset(gt_jsonfile)
	#read annotator system res
	df_hashtagger = read_testset(ht_jsonfile)
	voc_set = get_vocset(df_gt)
	df_rev_hashtagger = build_rev_word_index(df_hashtagger,voc_set)
	df_rev_gt = build_rev_word_index(df_gt, voc_set)
	recall_list = list()
	precision_list = list()
	nzr = 0
	for tag in voc_set:
		w_auto = set(df_rev_hashtagger.loc[df_rev_hashtagger.tag == tag]['item_lists'].values[0])
		w_h = set(df_rev_gt.loc[df_rev_gt.tag == tag]['item_lists'].values[0])
		w_c = w_auto.intersection(w_h)
		if(len(w_auto)==0):
			continue
		if(len(w_c) == 0):
			nzr = nzr + 1
		recall = len(w_c)/len(w_auto)
		precision = len(w_c)/len(w_h)
		recall_list.append(recall)
		precision_list.append(precision)

	precision = sum(precision_list)/len(voc_set)
	recall = sum(recall_list)/len(voc_set)
	ret = dict();
	ret['precision'] = precision
	ret['recall'] = recall
	ret['nzr'] = nzr
	return ret

def main():
	parser = argparse.ArgumentParser(description='Evaluate automatic image tagging.')
	parser.add_argument('gt_jsonfilename', action="store", help='filename of the json containing the groundtruth for the test set', type=str)
	parser.add_argument('ht_jsonfilename', action="store", help='filename of the json with the automatic annotations of the testset (to evaluate)')
	args = parser.parse_args()
	ret = eval(args.gt_jsonfilename, args.ht_jsonfilename)
	print(ret)

if __name__ == "__main__":
   main()

