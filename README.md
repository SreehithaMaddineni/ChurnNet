
#   Customer‑Churn‑Predictor

This project uses a basic artificial neural network (ANN) built in Python to analyze customer data and predict whether a customer is likely to churn. By identifying at-risk customers early, businesses can take proactive steps to improve customer retention, enhance satisfaction, and reduce revenue loss. The model is trained on real-world telecom data and demonstrates the practical value of machine learning in customer relationship management.

---

##  Why You Might Care

- **Telecoms & ISPs** – offer loyalty perks to subscribers right before they switch providers.  
- **Streaming & SaaS apps** – re‑engage users who look like they’re about to cancel.  
- **Banks & insurers** – keep valuable account holders from calling it quits.  

Basically, if you have customers, churn prediction can save you money and prevent losses.

---

##   How It Works 

1. **Load the data**  
   We start with Telco customer info (gender, services, monthly charges, etc.).

2. **Clean it up**  
   - Remove the customer ID (just a label).  
   - Turn Yes/No answers into 1/0.  
   - One‑hot‑encode things like “InternetService = DSL / Fiber / None”.  
   - Scale numbers so they’re all between 0 and 1.

3. **Look around**  
   Two quick histograms show how tenure and monthly charges differ between loyal folks and churners.

4. **Build a tiny neural net**  
   - 26 neurons → 10 neurons → 1 neuron (sigmoid).  
   - Train for 50 epochs.  
   - Optimizer? Good old SGD.

5. **Test it**  
   We hold out 20 % of the data and see how well the model predicts churn.  
   Typical accuracy: **~80 %**.

6. **See the results**  
   - A confusion‑matrix heat‑map.  
   - A neat precision / recall / F1 print‑out.  

---

##   Project Structure

```text
Customer_Churn_using_ANN/
├── Customer_Churn.py          # main script
├── Telco_customer_churn.xlsx  # dataset (or use the Kaggle CSV)
├── requirements.txt           # pip install stuff
└── README.md                  # you’re reading it!
```

---

##  Quickstart

```bash
git clone https://github.com/<your‑user>/Customer_Churn_using_ANN.git
cd Customer_Churn_using_ANN
pip install -r requirements.txt         # pandas, numpy, sklearn, tensorflow, ...
python Customer_Churn.py
```

> **Tip:** If you want to use the `.xlsx` file, keep it next to the script or change the path in `Customer_Churn.py`.

---

##  What You’ll See

| Step | You get |
|------|---------|
| Training | 50 lines of epoch logs  |
| Evaluation | `Test accuracy: 0.79` (ish) |
| Report | Precision / recall table |
| Plot | Confusion matrix heat‑map |



## Future Ideas

- The dataset is a bit imbalanced (fewer churners), so recall for churn may lag.  
- Try class weighting or SMOTE to balance things up.  
- Swap in a deeper net or a tree‑based model (XGBoost) to squeeze out more accuracy.



##  Author

**Sreehitha Maddineni** – 3rd‑year B.Tech (Math & Computing),IIT Ropar 




## License

MIT. Do what you want, just give credit. Enjoy!
