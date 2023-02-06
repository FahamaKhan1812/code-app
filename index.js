const express = require("express");

const app = express();
const port = 3000;

app.get("/lab14", (req, res) => {
  res.format({
    text: function () {
      res.send(
        `
            -------------------------------
                LAB#14-TASK NO 1
            -------------------------------
                #include<iostream>
            #include<algorithm>
            #include <vector>
            using namespace std;
            
            struct Token {
                char op;
                int location;
            };
            
            vector<Token> expr;
            
            // Function to add tokens to the expression
            void addToken(char op, int location) {
                Token token;
                token.op = op;
                token.location = location;
                expr.push_back(token);
            }
            
            // Function to sort the tokens based on precedence and location
            bool compare(Token& token1, Token& token2) {
                if (token1.op == '*' || token1.op == '/') {
                    if (token2.op == '+' || token2.op == '-')
                        return true;
                }
                else if (token1.op == '+' || token1.op == '-') {
                    if (token2.op == '*' || token2.op == '/')
                        return false;
                }
                return (token1.location < token2.location);
            }
            
            int main() {
                string exprString = "a/b*c-d*e+f";
                int len = exprString.length();
                for (int i = 0; i < len; i++) {
                    if (exprString[i] == '*' || exprString[i] == '/' ||
                        exprString[i] == '+' || exprString[i] == '-')
                        addToken(exprString[i], i);
                }
                sort(expr.begin(), expr.end(), compare);
                // Now, expr vector contains the sorted tokens
                // Now we can use this vector to generate the three-address code
                int temp_var_count = 0;
                for (int i = 0; i < expr.size(); i++) {
                    char op = expr[i].op;
                    int location = expr[i].location;
                    if (op == '*' || op == '/') {
                        // Generate three-address code for multiplication or division
                        char t1 = exprString[location - 1];
                        char t2 = exprString[location + 1];
                        cout << "t" << temp_var_count << " = " << t1 << " " << op << " " << t2 << endl;
                        temp_var_count++;
                    }
                    else if (op == '+' || op == '-') {
                        // Generate three-address code for addition or subtraction
                        char t1 = exprString[location - 1];
                        char t2 = exprString[location + 1];
                        cout << "t" << temp_var_count << " = " << t1 << " " << op << " t" << temp_var_count - 1 << endl;
                        temp_var_count++;
                    }
                }
                cout<<"result = t" << temp_var_count-1;
                return 0;
            }

            
            -------------------------------
                LAB#14-TASK NO 2
            -------------------------------
            #include <iostream>
            #include <sstream>
            #include <stack>
            #include <string>
            #include <vector>
            using namespace std;



            vector<string> infix_to_statements(const string &expression) {
                vector<string> statements;
                stack<string> operand_stack;
                stack<string> operator_stack;
                istringstream iss(expression);
                string token;
                while (iss >> token) {
                    if (token == "+" || token == "-" || token == "*" || token == "/") {
                        operator_stack.push(token);
                    } else if (token == "(") {
                        continue;
                    } else if (token == ")") {
                        string operand2 = operand_stack.top();
                        operand_stack.pop();
                        string operand1 = operand_stack.top();
                        operand_stack.pop();
                        string op = operator_stack.top();
                        operator_stack.pop();
                        string statement = operand1 + " " + op + " " + operand2 + ";";
                        statements.push_back(statement);
                        operand_stack.push("result");
                    } else {
                        operand_stack.push(token);
                    }
                }
                return statements;
            }

            string statement_to_assembly(string statement) {
                string assembly;
                istringstream iss(statement);
                string op1, op, op2;
                iss >> op1 >> op >> op2;

                if (op == "+") {
                    assembly = "movl " + op1 + ", %eax\n";
                    assembly += "addl " + op2 + ", %eax\n";
                } else if (op == "-") {
                    assembly = "movl " + op1 + ", %eax\n";
                    assembly += "subl " + op2 + ", %eax\n";
                } else if (op == "*") {
                    assembly = "movl " + op1 + ", %eax\n";
                    assembly += "imull " + op2 + ", %eax\n";
                } else if (op == "/") {
                    assembly = "movl " + op1 + ", %eax\n";
                    assembly += "movl " + op2 + ", %ebx\n";
                    assembly += "cltd\n";
                    assembly += "idivl %ebx\n";
                }
                return assembly;
            }

            int main() {
                string expression = "( a - b ) + ( a - c ) + ( a - c )";
                vector<string> statements = infix_to_statements(expression);

                for (vector<string>::iterator it = statements.begin(); it != statements.end(); it++) {
                    cout << *it << endl;
                    cout << statement_to_assembly(*it) << endl;
                
                }
                return 0;
            }



                `
      );
    },
  });
});

app.get("/stm", (req, res) => {
  res.format({
    text: function () {
      res.send(
        `
        /*
Fahama Khan
19B-039-CS-A
*/
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct Value
{
  std::string type;
  std::string Scope;
  int lineNumber;
};

bool DuplicationCheck(const std::string &name, const std::vector<std::map<std::string, Value>> &list)
{
  for (const auto &dictionary : list)
  {
    if (dictionary.count(name) > 0)
    {
      return true;
    }
  }
  return false;
}

void InsertInSTM(const std::string &name, const std::string &type, const std::string &Scope, const int &lineNumber,
                 std::vector<std::map<std::string, Value>> &list)
{
  if (!DuplicationCheck(name, list))
  {
    std::map<std::string, Value> dictionary;
    dictionary[name] = {type, Scope, lineNumber};
    list.push_back(dictionary);
  }
}

void printList(const std::vector<std::map<std::string, Value>> &list)
{
  std::cout << "Symbol table Manager:\n"
            << std::endl;
  std::cout << "------------------------------------------------------------" << std::endl;
  std::cout << "Name\t\t|Type\t\t|Scope\t\t|Line Number" << std::endl;
  std::cout << "------------------------------------------------------------" << std::endl;

  for (const auto &dictionary : list)
  {

    for (const auto &[name, value] : dictionary)
    {
      std::cout << name << "\t\t|" << value.type << "\t\t|" << value.Scope << "\t\t|" << value.lineNumber << std::endl;
    }
  }
  std::cout << "------------------------------------------------------------" << std::endl;
}

bool LookUp(const std::vector<std::map<std::string, Value>> &list,
            const std::string &name,
            Value &result)
{
  for (const auto &dictionary : list)
  {
    auto it = dictionary.find(name);
    if (it != dictionary.end())
    {
      result = it->second;
      return true;
    }
  }
  return false;
}

int main()
{
  std::vector<std::map<std::string, Value>> list;
  InsertInSTM("i", "int", "local", 5, list);
  InsertInSTM("j", "float", "local", 6, list);
  InsertInSTM("count", "const", "global", 7, list);
  InsertInSTM("help", "var", "local", 8, list);
  printList(list);

  Value result;
  std::string query = "j";

  if (LookUp(list, query, result))
  {
    std::cout << "\nFound:" << std::endl;
    std::cout << query << ": " << result.type << ", " << result.Scope << " --> " << result.lineNumber << std::endl;
  }
  else
  {
    std::cout << "Variable not found" << std::endl;
  }

  return 0;
}

        `
      );
    },
  });
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
