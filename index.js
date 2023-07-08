const express = require("express");

const app = express();
const port = 3000;

app.get("/help", (req, res) => {
  res.format({
    text: function () {
      res.send(
        `	

 	hhehe
         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
