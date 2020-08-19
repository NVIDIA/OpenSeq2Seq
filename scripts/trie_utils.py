import kenlm
#model_airtel=kenlm.Model('/nfs/alldata/Airtel/Manifest/pipeline/lm_out/full_data_text_only_v27.arpa')
import enchant
d = enchant.Dict("en_US")
white_list=['lte','dth']
class TrieNode:   
    def __init__(self): 
        self.children = [None]*27
        self.isEndOfWord = False
        self.mapped_phrase = None
        self.has_phrases = False
        
class Trie: 
    '''
    This class acts as a map dictionary.
    ''' 
    def __init__(self): 
        self.root = self.getNode() 
  
    def getNode(self): 
        return TrieNode() 
  
    def _charToIndex(self,ch): 
        if(ch!=' '):
            return ord(ch)-ord('a') 
        else:
            return 26
      
    def insert(self,key,mapped_phrase):

        '''
        key is the string to which something is to be mapped with
        mapped_phrase is the string mapped to key
        ''' 
        pCrawl = self.root 
        length = len(key) 
        for level in range(length): 
            index = self._charToIndex(key[level]) 
            if not pCrawl.children[index]: 
                pCrawl.children[index] = self.getNode() 
            pCrawl = pCrawl.children[index] 
            if(level!=length-1 and key[level+1] == ' '):
                pCrawl.isEndOfWord = True
                pCrawl.has_phrases = True
        pCrawl.mapped_phrase = mapped_phrase 
        pCrawl.isEndOfWord = True 
    
    def search(self, pCrawl, key):
        '''
        search key ( a string ) in the dictionary with "pCrawl" as the starting point
        '''   
        length = len(key) 
        for level in range(length): 
            index = self._charToIndex(key[level]) 
            if not pCrawl.children[index]: 
                return (None,None)
            pCrawl = pCrawl.children[index] 
  
        if(pCrawl != None and pCrawl.isEndOfWord):
            if(pCrawl.has_phrases):
                return (pCrawl.children[26], pCrawl.mapped_phrase)
            else:
                return (None, pCrawl.mapped_phrase)
        else:
            return (None,None)
       
    def search_children(self, index, words):
        N = len(words)
        i = index
        root = self.root
        while(i<N):
            x, y = self.search(root, words[i])
            if(x==None and y==None):
                return None, None
            elif(x!=None and y==None):
                root = x
            elif(x==None and y!=None):
                return i, y
            elif(x!=None and y!=None and i==N-1):
                return i, y
            else:
                root = x
            i+=1
        
        return None, None

    def check_support(self,word,replace_with,phrase1,phrase2,ngram_dict):
        if(word in white_list):
            return False
        support_of_orig=ngram_dict[0].get(word)
        if(support_of_orig==None):
            support_of_orig=0
        support_phrase_1_list=sorted(phrase1.split())
        support_phrase_1='_'.join(support_phrase_1_list)
        ngram_len_1=min(5,len(support_phrase_1_list))
        ngram_dict1=ngram_dict[ngram_len_1-1]
        support1=ngram_dict1.get(support_phrase_1)
        if(support1==None):
            support1=0

        support_phrase_2_list=sorted(phrase2.split())
        support_phrase_2='_'.join(support_phrase_2_list)
        support2=ngram_dict1.get(support_phrase_2)
        if(support2==None):
            support2=0
        # if(model_airtel.score(phrase2)>model_airtel.score(phrase1) and support1<=3):#if correct has zero support
        #         support2=support1+1
        if( (support2>support1+50 and support_of_orig<50) and len(word)!=2 and d.check(replace_with)==True  ): #if support of original word is very more than dont replace(50)

            # print("row",)
            print(support1,support2)
            print(phrase1,'replaced with',phrase2)
            return True
        return False



    def replace_sentence(self, sentence,dict_ngram={},opt='all'):
        '''
        This function replaces all of the source strings to target strings ( mappings inserted into the Trie )
        Constraints:
        The input string must be have characters from 'a' to 'z' or ' '(space character)
        Time complexity is O(length of the input string)
        '''
        words = sentence.split()
        N = len(words)
        replaced_sentence = ""
        i = 0
        while(i<N):
            j, y = self.search_children(i, words)
            if(y!=None):
                if(opt!='all'):
                    phrase_without_replace = ' '.join(words[max(0,i-1):min(N,i+2)])
                    if(i!=0):
                        left = words[i-1]
                    else:
                        left = ""
                    if(i!=N-1):
                        right = words[i+1]
                    else:
                        right = ""
                    phrase_with_replace = left + ' '+y +' '+ right

                    if(self.check_support(words[i],y,phrase_without_replace, phrase_with_replace, dict_ngram)==False):
                        y = words[i]
                replaced_sentence += y+" "
                i=j+1
            else:
                replaced_sentence += words[i]+" "
                i+=1
        return replaced_sentence[:-1]
