#test commit

def caesarCipher(s, k):
    #deal with uppercase
    print(s)
    upper_inds =[index for index, char in enumerate(s) if char.isupper()]
    print("these are upper inds", upper_inds)
    s = s.lower()
    print("this is s",s)
    
    s = list(s)
    alpha_char = set("thequickbrownfoxjumpsoverthelazydog")
    
    
    
    #specials
    specials = {char: [i for i, j in enumerate(s) if j == char] for char in s if char not in alpha_char}
    print("these are speicals",specials)
    for i in specials:
        s = [s[x] for x,p in enumerate(s) if x not in specials[i]]

    print("this is s with specials removed",s)
    

        
    alphabet = list(sorted(alpha_char))
    standard_dict_char = {alphabet[i]: i for i in range(len(alphabet))}
    standard_dict_number = {i: alphabet[i] for i in range(len(alphabet))}
    print(standard_dict_char, standard_dict_number)
  
    out_list = []
    for char in s:
        out_list.append(standard_dict_number[(standard_dict_char[char]+k)%26])
        print(out_list)
    
    print(out_list)
    #readd specialsl
    for i in  specials:
        for j in specials[i]:
            out_list.insert(j,i)  
    
    
    for i in upper_inds:
        out_list[i] = out_list[i].upper()
    print(out_list)
    print("hello","".join(out_list))

# caesarCipher("Always-Look-on-the-Bright-Side-of-Life",5)
caesarCipher("1X7T4VrCs23k4vv08D6yQ3S19G4rVP188M9ahuxB6j1tMGZs1m10ey7eUj62WV2exLT4C83zl7Q80M",27)