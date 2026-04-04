# 9.7.2 Extended-Precision Integer Arithmetic Instructions

Documents PTX instructions for extended-precision (multi-word) integer arithmetic using carry/borrow flags: `add.cc` (add with carry-out), `addc` (add with carry-in/out), `sub.cc` (subtract with borrow-out), `subc` (subtract with borrow-in/out), `mad.cc` (multiply-add with carry-out), and `madc` (multiply-add with carry-in/out). Used to implement 128-bit and wider arithmetic.
