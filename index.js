const express = require("express");

const app = express();
const port = 3000;

app.get("/", (req, res) => {
  res.send(
    `
-------------------------------
    TASK NO 1
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
    `
  );
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
