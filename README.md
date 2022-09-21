# CPP-Notes
These are the notes for CSE201


<details><summary>Lecture 1</summary>
<p>

Course structue: 

* 7 weeks - programming and software engineering
* 6 weeks - collaborative project 

```cpp
   // Your First C++ Program

  #include <iostream>

  int main() {
      std::cout << "Hello World!";
      return 0;
  }
```
   
## References and Pointers

- variables stored in memory, at a given **address**
- they *have* an **address** and *contain* a **value**



- a *reference* is a **supplementary name for a variable**
- syntax: `typeName& referenceName=variableName;`

```cpp
unsigned char i=165; // The variable i is stored at address 17
unsigned char& r=i; // r is just **another name** for i
```

- a *pointer* is a variable that **contains the address of another variable**
- syntax: `typeName* pointerName=&variableName;`

```cpp
unsigned char i=165; // The variable i is stored at address 17
unsigned char* r=&i; // The variable p contains the value 17, p is stored elsewhere
```

- p **points** to the variable i
- \*p **depoints** p

```cpp
unsigned char* p; // The variable p contains the address of i 
unsigned char *p; // The variable *p contains the value at the address p, so i
```

- you can do pointer arithmetics
- `p++;` adds the memory space of the object (moves to next object)


- A variable representing an array **"is" a pointer to the first element of the array**

```cpp
unsigned char i[10]; // i is a pointer to the first element
unsigned char* p=i; // p is equal to i
```
- p moves within the array (moves by memory length of `unsigned char`)

```cpp
*(p+4)=65; // affects 65 to the 5th element of the array
std::cout << i[4] << std:endl; // prints A (char corresponding to 65)
```
- when the program terminates, all allocated memory is cleared

## Tests

- Attention: `==` vs. `=`
```cpp
if(i==2){
    // enter if i is equal to 2
}
if(i=2){
    // i is now 2
    // 2 is converted to True, always enter statement
}
```

- `&&` : and
- `||` : or
- `=!` : not

</p>
</details>
