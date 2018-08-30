def test():
	raise IOError("Doesn't work") 
	print('hahahah')
	return 9

a = 0
a = test()
print(a)
