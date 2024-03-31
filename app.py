import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt



with st.sidebar:
    st.image("https://lapasseduvent.com/wp-content/uploads/2022/08/meme-stonks.jpg")
    st.title("Portfolio Optimization Tool")
    choice = st.radio("Steps", ["Data Exploration", "Efficient Frontier", "Optimal portfolio"])
    st.info("This tool uses synthetic data and Markowitz Mean-Variance optimization to select optimal portfolios.")

tickers = ['Apple', 'Microsoft', 'Netflix', 'Novartis']

mu = np.array([0.05, 0.06, 0.08, 0.06])
vol = np.array([0.15, 0.2, 0.25, 0.3])
data = pd.DataFrame(data={'mu' : mu, 'vol' : vol} , index= tickers)

correl_matrix = np.array([[1, 0.1, 0.4, 0.5], [0.1, 1, 0.7, 0.4], [0.4, 0.7, 1, 0.8], [0.5, 0.4, 0.8, 1]])
x0 = np.array([0.25, 0.25, 0.25, 0.25])



if choice == "Data Exploration":
    st.info("This page explores our data")

    st.write("Expected returns")
    st.bar_chart(data['mu'])
    st.write("Expected volatilities")
    st.bar_chart(data['vol'])

    st.write("Correlation matrix")
    correl_matrix = np.array([[1, 0.1, 0.4, 0.5], [0.1, 1, 0.7, 0.4], [0.4, 0.7, 1, 0.8], [0.5, 0.4, 0.8, 1]])
    plot = sns.heatmap(pd.DataFrame(correl_matrix, index = tickers, columns = tickers), annot=True)
    st.pyplot(plot.get_figure())


cov_matrix = correl_matrix * (vol.reshape(1,-1).T @ vol.reshape(1,-1))

def QP(x, sigma, mu, gamma):
    v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu
    return v

def efficient_frontier(gam, constraints):
    res = minimize(QP, x0, args=(cov_matrix, mu, gam), options={'disp': False}, constraints=constraints)
    optimized_weights = res.x
    mu_optimized = optimized_weights @ mu
    vol_optimized = np.sqrt(optimized_weights @ cov_matrix @ optimized_weights)

    return mu_optimized, vol_optimized, optimized_weights

# Fully invested weight constraint
constraints = [LinearConstraint(np.ones(x0.shape), ub = 1), LinearConstraint(-np.ones(x0.shape), ub = -1)]


if choice == "Efficient Frontier":
    st.info("This page shows the efficient frontier based on data and a user-entered risk-free rate")
    mu_efficient_frontier = []
    vol_efficient_frontier = []
    weights_efficient_frontier = []

    for gam in np.linspace(-0.5, 3, 101):
        mu_efficient_frontier.append(efficient_frontier(gam, constraints=constraints)[0])
        vol_efficient_frontier.append(efficient_frontier(gam, constraints=constraints)[1])
        weights_efficient_frontier.append(efficient_frontier(gam, constraints=constraints)[2])

    chart_data = pd.DataFrame( data = {'vol':vol_efficient_frontier, 'mu': mu_efficient_frontier, 'color' : '#0000FF', 'size' : 2})

    rf = st.slider('Risk free rate?', 0.0, 0.04, step = 0.01)
    st.session_state['rf'] = rf       # saves value of rf for other pages

    adding_rf = {'vol':0 , 'mu': rf, 'color':'#FFA500','size' : 10}
    chart_data = pd.concat([chart_data, pd.DataFrame([adding_rf])], ignore_index=True)

    weights_tangency = (np.linalg.inv(cov_matrix) @ (mu - rf)) / (np.ones(cov_matrix.shape[0]) @ np.linalg.inv(cov_matrix) @ (mu - rf))
    st.session_state['weights_tangency'] = weights_tangency
    mu_tangency = weights_tangency @ mu
    vol_tangency = np.sqrt(weights_tangency @ cov_matrix @ weights_tangency )
    adding_tangency = {'vol': vol_tangency, 'mu': mu_tangency, 'color': '#FF0000','size' : 10}
    chart_data = pd.concat([chart_data, pd.DataFrame([adding_tangency])], ignore_index=True)


    st.scatter_chart(
        chart_data,
        x='vol',
        y='mu',
        color='color',
        size = 'size'
    )


if choice == "Optimal portfolio":
    st.info("This page shows the efficient frontier based on data, the user-entered risk-free rate and gamma")
    if 'rf' in st.session_state :
        rf = st.session_state.rf # get rf back from saved variable
        weights_tangency = st.session_state.weights_tangency  # get weights_tangency back from saved variable

        mu_mod = np.append(mu, rf)
        vol_mod = np.append(vol, 0)
        cov_matrix_mod = np.zeros([cov_matrix.shape[0] + 1, cov_matrix.shape[0] + 1])  # create increased size covariance matrix
        cov_matrix_mod[:-1,:-1] = cov_matrix  # fill all rows and all columns excecpt last ones with previous cov matrix
        x0_mod = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # increase the size by 1 to include risk-free as last element

        on = st.toggle('Short selling allowed')
        gam = st.slider('Gamma?', 0.0, 1.0, step=0.1)

        if on:
            constraints = [LinearConstraint(np.ones(x0_mod.shape), ub=1),
                           LinearConstraint(-np.ones(x0_mod.shape), ub=-1)]

            res = minimize(QP, x0_mod, args=(cov_matrix_mod, mu_mod, gam), options={'disp': False},
                           constraints=constraints)
            optimized_weights = res.x
            mu_optimized = optimized_weights @ mu_mod
            vol_optimized = np.sqrt(optimized_weights @ cov_matrix_mod @ optimized_weights)
            st.bar_chart(pd.DataFrame(optimized_weights, index= tickers + ['risk-free']))

        else:

            constraints = [LinearConstraint(np.ones(x0_mod.shape), ub = 1), LinearConstraint(-np.ones(x0_mod.shape), ub = -1), LinearConstraint(np.eye(x0_mod.shape[0]), lb = 0)]

            res = minimize(QP, x0_mod, args=(cov_matrix_mod, mu_mod, gam), options={'disp': False}, constraints=constraints)
            optimized_weights = res.x
            mu_optimized = optimized_weights @ mu_mod
            vol_optimized = np.sqrt(optimized_weights @ cov_matrix_mod @ optimized_weights)

            fig1, ax1 = plt.subplots()
            ax1.pie(np.abs(optimized_weights), labels=tickers + ['risk-free'], autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig1)

    else:
        st.write('You must select the risk-free rate in the tab "Efficient frontier" first.')


